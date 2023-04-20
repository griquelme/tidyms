from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Type, Union
from pathlib import Path
from sqlalchemy import Column, Float, ForeignKey, Integer, String, Table, MetaData
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.exc import IntegrityError
from ..lcms import Annotation, Feature, Roi
from .. import _constants as c
import json


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

    def to_sample(self) -> Sample:
        return Sample(
            str(self.path), self.id, self.ms_level, self.start, self.end, self.group
        )


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

    def to_roi(self, roi_type: Type[Roi]) -> Roi:
        roi = roi_type.from_string(self.data)
        roi.id = self.id
        return roi


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
        Column(
            "id",
            Integer,
            ForeignKey("features.id"),
            primary_key=True,
            autoincrement=True,
        ),
        Column("sample_id", String, ForeignKey("samples.id")),
        Column("roi_id", Integer, ForeignKey("rois.id")),
        Column("label", Integer, ForeignKey("annotations.label")),
    ]

    descriptors = [Column(x, Float) for x in feature_type.descriptor_names()]
    columns.extend(descriptors)
    attrs = {
        "__tablename__": table_name,
        "__table__": Table(table_name, metadata, *columns),
    }

    return type("DescriptorModel", (Base,), attrs)


class AssayData:
    """
    Manages Sample, Roi and Feature persistence in an Assay using a SQLite DataBase.

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
        """
        Create the DescriptorModel table using descriptors available in the Feature class.

        Parameters
        ----------
        feature_type : Feature
            Feature type stored in the AssayData.

        """
        if not hasattr(cls, "DescriptorModel"):
            cls.DescriptorModel = _create_descriptor_table(feature_type, Base.metadata)

    def add_samples(self, samples: list[Sample]):
        """
        Add samples into the DB.

        Parameters
        ----------
        samples : list[Sample]

        Raises
        ------
        ValueError
            If a sample with the same id already exists in the DB.

        """
        samples = [SampleModel(**x.to_dict()) for x in samples]
        with self.SessionFactory() as session:
            try:
                session.add_all(samples)
                session.commit()
            except IntegrityError:
                msg = "Trying to insert a sample with an existing ID"
                raise ValueError(msg)

    def flag_processed_samples(self, samples: list[Sample], step: str):
        """
        Flags samples as processed in a preprocessing step.

        Parameters
        ----------
        samples : list[Sample]
        step : str
            Name of a preprocessing step.

        Raises
        ------
        ValueError
            If the preprocessing step name is not a valid name. Valid names
            are defined by `PREPROCESSING_STEPS` in the `_constants` module.

        """
        _check_preprocessing_step(step)
        samples = [ProcessedSampleModel(sample_id=x.id, step=step) for x in samples]
        with self.SessionFactory() as session:
            session.add_all(samples)
            session.commit()

    def get_samples(self, step: Optional[str] = None) -> list[Sample]:
        """
        Load the samples stored in the DB.

        Parameters
        ----------
        step : str or None, default=None
            Name of a preprocessing step. If provided, returns all samples processed
            by that processing step. If ``None``, return all samples in the DB.

        Returns
        -------
        list[Sample]

        Raises
        ------
        ValueError
            If the preprocessing step name is not a valid name. Valid names
            are defined by `PREPROCESSING_STEPS` in the `_constants` module.

        """
        _check_preprocessing_step(step)
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
                samples.append(row.SampleModel.to_sample())
        return samples

    def delete_samples(self, sample_ids: list[str]) -> None:
        """
        Delete samples stored in the DB.

        Parameters
        ----------
        sample_ids : list[str]
            Unique IDs of samples in the DB.

        """
        with self.SessionFactory() as session:
            stmt = delete(SampleModel).where(SampleModel.id.in_(sample_ids))
            session.execute(stmt)
            session.commit()

    def set_processing_parameters(self, step: str, parameters: dict):
        """
        Stores preprocessing parameters of a step in the DB.

        Parameters
        ----------
        step : str
            Preprocessing step name.
        parameters : dict
            Parameters used in the preprocessing step.

        Raises
        ------
        ValueError
            If the preprocessing step name is not a valid name. Valid names
            are defined by `PREPROCESSING_STEPS` in the `_constants` module.

        """
        _check_preprocessing_step(step)
        with self.SessionFactory() as session:
            param_str = json.dumps(parameters)
            params_model = ProcessParameterModel(step=step, parameters=param_str)
            session.add(params_model)
            session.commit()

    def get_processing_parameters(self, step: str) -> dict:
        """
        Retrieves preprocessing parameters of a step from the DB.

        Parameters
        ----------
        step : str
            Preprocessing step name.

        Returns
        -------
        parameters : dict
            Parameters used in the preprocessing step.

        Raises
        ------
        ValueError
            If the preprocessing step name is not a valid name. Valid names
            are defined by `PREPROCESSING_STEPS` in the `_constants` module.

        """
        _check_preprocessing_step(step)
        with self.SessionFactory() as session:
            stmt = select(ProcessParameterModel).where(
                ProcessParameterModel.step == step
            )
            results = session.execute(stmt)
            param_model = results.first()
            if param_model is None:
                params = dict()
            else:
                params = json.loads(param_model.ProcessParameterModel.parameters)
        return params

    def add_roi_list(self, roi_list: Sequence[Roi], sample: Sample):
        """
        Stores a list of ROI extracted from a sample in the DB.

        Parameters
        ----------
        roi_list : Sequence[Roi]
        sample : Sample

        """
        roi_model_list = list()
        for roi in roi_list:
            roi_model = RoiModel(sample_id=sample.id, data=roi.to_string())
            roi_model_list.append(roi_model)
        with self.SessionFactory() as session:
            session.add_all(roi_model_list)
            session.commit()

    def get_roi_list(self, sample: Sample) -> list[Roi]:
        """
        Retrieves a list of ROI detected in a sample.

        Parameters
        ----------
        sample : Sample

        Returns
        -------
        list[Roi]

        """
        with self.SessionFactory() as session:
            stmt = select(RoiModel).where(RoiModel.sample_id == sample.id)
            result = session.execute(stmt)
            roi_list = list()
            for row in result:
                roi = row.RoiModel.to_roi(self.roi)
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
                        sample_id=sample.id,
                        roi_id=roi.id,
                        label=ann.label,
                        **ft.describe(),
                    )

                    feature_model_list.append(ft_model)
                    annotation_model_list.append(annotation_model)
                    descriptor_model_list.append(descriptor_model)
        with self.SessionFactory() as session:
            session.add_all(feature_model_list)
            session.add_all(annotation_model_list)
            session.add_all(descriptor_model_list)
            session.commit()

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
                stmt = select(RoiModel).join(FeatureModel).where(cond)
            else:
                cond = RoiModel.sample_id == sample.id
                stmt = select(RoiModel).where(cond)

            result = session.execute(stmt)
            feature_list = list()
            for row in result:
                rm = row.RoiModel
                roi = rm.to_roi(self.roi)
                for ft_model, ann_model in zip(rm.features, rm.annotations):
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


def _check_preprocessing_step(step: Optional[str]):
    if (step is not None) and (step not in c.PREPROCESSING_STEPS):
        valid_steps = ", ".join(c.PREPROCESSING_STEPS)
        msg = (
            f"{step} is not a valid preprocessing step. valid values are {valid_steps}."
        )
        raise ValueError(msg)
