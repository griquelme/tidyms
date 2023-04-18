from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Type, Union
from pathlib import Path
from sqlalchemy import Column, Float, ForeignKey, Integer, String, Table, MetaData
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import declarative_base, sessionmaker, Mapped, mapped_column, relationship
from sqlalchemy.exc import IntegrityError
from ..lcms import Annotation, Feature, Roi
import json

# TODO: get sample by processing stage
# TODO: Idea: Assays are created using create_assay and load_assay functions.


@dataclass
class Sample:
    """
    Sample class to manage iteration over MSData objects.

    Attributes
    -------
    path : Path or str
        Path to a raw data file.
    id : str
        Sample name.
    ms_level : int, default=1
        MS level of data to use.
    start: float or None, default=None
        Minimum acquisition time of MS scans to include. If ``None``, start from the first scan.
    end: float or None, default=None
        Maximum acquisition time of MS scans to include. If ``None``, end at the last scan.
    group : str, default=""
        Sample group.

    """

    path: Union[Path, str]
    id: str
    ms_level: int = 1
    start: Optional[float] = None
    end: Optional[float] = None
    group: str = ""

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)

        if self.id is None:
            self.id = self.path.stem

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(d["path"])
        return d


Base = declarative_base()


class ProcessParameterModel(Base):
    """
    Model for the Table with processing parameters of the Assay.

    """

    __tablename__ = "parameters"
    step: Mapped[str] = mapped_column(String, primary_key=True)
    parameters: Mapped[str] = mapped_column(String)


class SampleModel(Base):
    """
    Model for Sample objects included in the Assay.

    """

    __tablename__ = "samples"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    path: Mapped[str] = mapped_column(String, nullable=False)
    ms_level: Mapped[int] = mapped_column(Integer)
    start: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    end: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    group: Mapped[str] = mapped_column(String, nullable=True)
    rois: Mapped[list["RoiModel"]] = relationship(back_populates="samples")
    features: Mapped[list["FeatureModel"]] = relationship(back_populates="samples")


class ProcessedSampleModel(Base):
    """
    Model to record Samples processed in the different preprocessing stages.

    """

    __tablename__ = "processed_samples"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_id: Mapped[str] = mapped_column(String, ForeignKey("samples.id"))
    step: Mapped[str] = mapped_column(String, ForeignKey("parameters.step"))


class RoiModel(Base):
    """
    Model for ROI extracted in each Sample.

    """

    __tablename__ = "rois"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_id: Mapped[str] = mapped_column(ForeignKey("samples.id"))
    data: Mapped[str] = mapped_column(String)
    samples: Mapped["SampleModel"] = relationship(back_populates="rois")
    features: Mapped[list["FeatureModel"]] = relationship(
        back_populates="roi", lazy="immediate"
    )
    annotations: Mapped[list["AnnotationModel"]] = relationship(
        back_populates="roi", lazy="immediate"
    )


class FeatureModel(Base):
    """
    Model for Features extracted in each ROI.

    """

    __tablename__ = "features"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_id: Mapped[str] = mapped_column(ForeignKey("samples.id"))
    roi_id: Mapped[int] = mapped_column(ForeignKey("rois.id"))
    data: Mapped[str] = mapped_column(String)
    roi: Mapped["RoiModel"] = relationship(back_populates="features")
    samples: Mapped["SampleModel"] = relationship(back_populates="features")
    annotation: Mapped["AnnotationModel"] = relationship(back_populates="feature")


class AnnotationModel(Base):
    """
    Model for annotations of each Feature.

    """

    __tablename__ = "annotations"
    id: Mapped[int] = mapped_column(
        ForeignKey("features.id"), primary_key=True, autoincrement=True
    )
    sample_id: Mapped[str] = mapped_column(ForeignKey("samples.id"))
    roi_id: Mapped[int] = mapped_column(ForeignKey("rois.id"))
    label: Mapped[int] = mapped_column(Integer)
    isotopologue_label: Mapped[int] = mapped_column(Integer)
    isotopologue_index: Mapped[int] = mapped_column(Integer)
    charge: Mapped[int] = mapped_column(Integer)
    roi: Mapped["RoiModel"] = relationship(back_populates="annotations")
    feature: Mapped["FeatureModel"] = relationship(back_populates="annotation")

    def to_annotation(self) -> Annotation:
        return Annotation(
            label=self.label,
            isotopologue_label=self.isotopologue_label,
            isotopologue_index=self.isotopologue_index,
            charge=self.charge,
        )


def _create_descriptor_table(feature_type: Type[Feature], metadata: MetaData):
    """
    Creates a Model for descriptors of features.

    The table is created using available descriptors from the Feature class.

    Parameters
    ----------
    feature_type : Type[Feature]
        The Feature class used in the Assay.
    metadata : MetaData
        DB Metadata.

    Returns
    -------
    DescriptorModel: Base

    """
    table_name = "descriptors"

    columns = [
        Column("id", Integer, ForeignKey("features.id"), primary_key=True, autoincrement=True),
        Column("sample_id", String, ForeignKey("samples.id")),
        Column("roi_id", Integer, ForeignKey("rois.id")),
        Column("label", Integer, ForeignKey("annotations.label")),
    ]

    descriptors = [Column(x, Float) for x in feature_type.descriptor_names()]
    columns.extend(descriptors)
    attrs = {"__tablename__": table_name, "__table__": Table(table_name, metadata, *columns)}

    return type("DescriptorModel", (Base,), attrs)


class AssayData:
    """
    Manages Sample, Roi and Feature persistence in an Assay.

    """

    DescriptorModel: Type

    def __init__(self, name: Union[Path, str], roi: Type[Roi], feature: Type[Feature]):

        self.roi = roi
        self.feature = feature

        # Create a database engine and session factory
        if name:
            db_address = f"sqlite:///{name}.db"
        else:
            db_address = "sqlite:///:memory:"
        self.engine = create_engine(db_address)
        self.SessionFactory = sessionmaker(bind=self.engine)
        self._create_descriptor_model(feature)

        # Create the table in the database
        Base.metadata.create_all(self.engine)

    @classmethod
    def _create_descriptor_model(cls, feature_type: Type):
        if not hasattr(cls, "DescriptorModel"):
            cls.DescriptorModel = _create_descriptor_table(feature_type, Base.metadata)

    def add_samples(self, samples: list[Sample]):
        samples = [SampleModel(**x.to_dict()) for x in samples]
        with self.SessionFactory() as session:
            try:
                session.add_all(samples)
                session.commit()
            except IntegrityError:
                msg = "Trying to insert a sample with an existing ID"
                raise ValueError(msg)

    def flag_processed(self, samples: list[Sample], step: str):
        samples = [ProcessedSampleModel(sample_id=x.id, step=step) for x in samples]
        with self.SessionFactory() as session:
            try:
                session.add_all(samples)
                session.commit()
            except IntegrityError:
                msg = "Trying to insert a sample with an existing ID"
                raise ValueError(msg)

    def get_samples(self, step: Optional[str] = None) -> list[Sample]:
        with self.SessionFactory() as session:
            if step is None:
                stmt = select(SampleModel)
            else:
                stmt = (
                    select(SampleModel)
                    .join(ProcessedSampleModel)
                    .where(ProcessedSampleModel.step == step)
                )
            result = session.execute(stmt)
            samples = list()
            for row in result:
                sm = row.SampleModel
                sample = Sample(Path(sm.path), sm.id, sm.ms_level, sm.start, sm.end, sm.group)
                samples.append(sample)
        return samples

    def delete_samples(self, sample_ids: list[str]) -> None:
        with self.SessionFactory() as session:
            stmt = delete(SampleModel).where(SampleModel.id.in_(sample_ids))
            session.execute(stmt)
            session.commit()

    def set_processing_parameters(self, step: str, parameters: dict):
        with self.SessionFactory() as session:
            param_str = json.dumps(parameters)
            params_model = ProcessedSampleModel(step=step, parameters=param_str)
            session.add(params_model)
            session.commit()

    def get_processing_parameters(self, step: str) -> dict:
        with self.SessionFactory() as session:
            stmt = select(ProcessParameterModel).where(ProcessParameterModel.step == step)
            results = session.execute(stmt)
            param_model = results.first()
            if param_model is None:
                params = dict()
            else:
                params = json.loads(param_model.ProcessParameterModel.parameters)
        return params

    def add_roi_list(self, roi_list: Sequence[Roi], sample: Sample):

        roi_model_list = list()
        for roi in roi_list:
            roi_model = RoiModel(sample_id=sample.id, data=roi.to_string())
            roi_model_list.append(roi_model)
        with self.SessionFactory() as session:
            try:
                session.add_all(roi_model_list)
                session.commit()
            except IntegrityError:
                msg = "Trying to insert a sample with an existing ID"
                raise ValueError(msg)

    def get_roi_list(self, sample: Sample) -> list[Roi]:
        with self.SessionFactory() as session:
            stmt = select(RoiModel).where(RoiModel.sample_id == sample.id)
            result = session.execute(stmt)
            roi_list = list()
            for row in result:
                rm = row.RoiModel
                roi = self.roi.from_string(rm.data)
                roi.id = rm.id
                roi_list.append(roi)
        return roi_list

    def delete_roi_list(self, sample: Sample):
        with self.SessionFactory() as session:
            stmt = delete(RoiModel).where(RoiModel.sample_id == sample.id)
            session.execute(stmt)
            session.commit()

    def add_features(self, roi_list: Sequence[Roi], sample: Sample):
        feature_model_list = list()
        annotation_model_list = list()
        descriptor_model_list = list()
        for roi in roi_list:
            if roi.features is not None:
                for ft in roi.features:
                    ft_model = FeatureModel(
                        sample_id=sample.id, roi_id=roi.id, data=ft.to_str()
                    )
                    ann = ft.annotation
                    annotation_model = AnnotationModel(
                        sample_id=sample.id,
                        roi_id=roi.id,
                        label=ann.label,
                        charge=ann.charge,
                        isotopologue_label=ann.isotopologue_label,
                        isotopologue_index=ann.isotopologue_index,
                    )

                    descriptor_model = self.DescriptorModel(
                        sample_id=sample.id, roi_id=roi.id, label=ann.label, **ft.describe()
                    )

                    feature_model_list.append(ft_model)
                    annotation_model_list.append(annotation_model)
                    descriptor_model_list.append(descriptor_model)
        with self.SessionFactory() as session:
            try:
                session.add_all(feature_model_list)
                session.add_all(annotation_model_list)
                session.add_all(descriptor_model_list)
                session.commit()
            except IntegrityError:
                msg = "Trying to insert a sample with an existing ID"
                raise ValueError(msg)

    def get_features(
        self,
        sample: Optional[Sample] = None,
        label: Optional[int] = None,
        groups: Optional[list[str]] = None,
    ) -> list[Feature]:
        with self.SessionFactory() as session:

            if sample is None:
                if label is None:
                    msg = "`sample` or `label` must be specified to select features"
                    raise (ValueError(msg))
                cond = AnnotationModel.label == label
                if groups is not None:
                    cond = cond and (SampleModel.group.in_(groups))
            else:
                cond = RoiModel.sample_id == sample.id

            stmt = select(RoiModel).where(cond)
            result = session.execute(stmt)
            feature_list = list()
            for row in result:
                roi_model = row.RoiModel
                roi = self.roi.from_string(roi_model.data)
                for ft_model, ann_model in zip(roi_model.features, roi_model.annotations):
                    ann = ann_model.to_annotation()
                    ft = self.feature.from_str(ft_model.data, roi, ann)
                    feature_list.append(ft)
        return feature_list

    def get_descriptors(
        self, sample: Optional[Sample] = None, descriptors: Optional[list[str]] = None
    ):
        if descriptors is None:
            descriptors = self.feature.descriptor_names()
        else:
            for d in descriptors:
                valid_descriptors = self.feature.descriptor_names()
                if d not in valid_descriptors:
                    valid_str = ", ".join(valid_descriptors)
                    msg = f"{d} is not a valid descriptor. Valid descriptors are: {valid_str}."
                    raise ValueError(msg)

        descriptors.extend(["label", "sample_id", "id"])
        descriptor_list = {x: list() for x in descriptors}

        with self.SessionFactory() as session:
            if sample is None:
                stmt = select(self.DescriptorModel)
            else:
                stmt = select(self.DescriptorModel).where(
                    self.DescriptorModel.sample_id == sample.id
                )
            result = session.execute(stmt)
            for row in result:
                for d in descriptors:
                    descriptor_list[d].append(row.DescriptorModel.__dict__[d])
        return descriptor_list
