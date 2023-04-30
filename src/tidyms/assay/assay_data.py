"""
Storage tools for the Assay class.

"""


import json
import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import delete, select, Column, Float, ForeignKey, Integer, String
from sqlalchemy.exc import IntegrityError
from ..lcms import Annotation, Feature, Roi
from .. import _constants as c
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence, Type, Union


# TODO: test delete all ROI.
# TODO : test get_unprocessed_samples.


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
    start_time: float, default=0.0
        Minimum acquisition time of MS scans to include. If ``None``, start from the first scan.
    end_time: float or None, default=None
        Maximum acquisition time of MS scans to include. If ``None``, end at the last scan.
    group : str, default=""
        Sample group.

    """

    path: Union[Path, str]
    id: str
    ms_level: int = 1
    start_time: float = 0.0
    end_time: Optional[float] = None
    group: str = ""

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(d["path"])
        return d


Base = orm.declarative_base()


class ProcessParameterModel(Base):
    """
    Model for the Table with processing parameters of the Assay.

    """

    __tablename__ = "parameters"
    step: Mapped[str] = mapped_column(sa.String, primary_key=True)
    parameters: Mapped[str] = mapped_column(sa.String)


class SampleModel(Base):
    """
    Model for Sample objects included in the Assay.

    """

    __tablename__ = "samples"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    path: Mapped[str] = mapped_column(String, nullable=False)
    ms_level: Mapped[int] = mapped_column(Integer)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    group: Mapped[str] = mapped_column(String, nullable=True)
    # rois: Mapped[list["RoiModel"]] = relationship(back_populates="samples")
    # features: Mapped[list["FeatureModel"]] = relationship(back_populates="samples")

    def to_sample(self) -> Sample:
        return Sample(
            str(self.path), self.id, self.ms_level, self.start_time, self.end_time, self.group
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
    sample_id: Mapped[str] = mapped_column(ForeignKey("samples.id"), index=True)
    data: Mapped[str] = mapped_column(String)
    # samples: Mapped["SampleModel"] = relationship(back_populates="rois")
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
    # sample_id: Mapped[str] = mapped_column(ForeignKey("samples.id"))
    roi_id: Mapped[int] = mapped_column(ForeignKey("rois.id"))
    data: Mapped[str] = mapped_column(String)
    roi: Mapped["RoiModel"] = relationship(back_populates="features", lazy="immediate")
    # samples: Mapped["SampleModel"] = relationship(back_populates="features", lazy="immediate")
    annotation: Mapped["AnnotationModel"] = relationship(
        back_populates="feature", lazy="immediate"
    )

    def to_feature(
        self, feature_type: Type[Feature], roi: Roi, annotation: Annotation
    ) -> Feature:
        return feature_type.from_str(self.data, roi, annotation)


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


def _create_descriptor_table(feature_type: Type[Feature], metadata: sa.MetaData):
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

    columns: list[Column] = [
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
        "__table__": sa.Table(table_name, metadata, *columns),
    }

    return type("DescriptorModel", (Base,), attrs)


class AssayData:
    """
    Manages Sample, Roi and Feature persistence in an Assay using a SQLite DataBase.

    """

    DescriptorModel: Type

    def __init__(
        self,
        name: Union[Path, str],
        roi: Type[Roi],
        feature: Type[Feature],
        echo: bool = False,
    ):
        self.roi = roi
        self.feature = feature

        # Create a database engine and session factory
        if name:
            db_address = f"sqlite:///{name}"
        else:
            db_address = "sqlite:///:memory:"
        self.engine = sa.create_engine(db_address, echo=echo)
        self.SessionFactory = orm.sessionmaker(bind=self.engine)
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
        with self.SessionFactory() as session:
            try:
                session.add_all([SampleModel(**x.to_dict()) for x in samples])
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
        processed = [ProcessedSampleModel(sample_id=x.id, step=step) for x in samples]
        with self.SessionFactory() as session:
            session.add_all(processed)
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
            stmt = select(SampleModel)
            if step is not None:
                stmt = stmt.join(ProcessedSampleModel).where(ProcessedSampleModel.step == step)
            result = session.execute(stmt)
            samples = list()
            for row in result:
                samples.append(row.SampleModel.to_sample())
        return samples

    def get_unprocessed_samples(self, step: str) -> list[Sample]:
        all_samples_dict = {x.id: x for x in self.get_samples()}
        processed_samples = self.get_samples(step)
        for sample in processed_samples:
            all_samples_dict.pop(sample.id)
        unprocessed_samples = list(all_samples_dict.values())
        return list(unprocessed_samples)

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

    def get_processing_parameters(self, step: str) -> Optional[dict]:
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
            stmt = select(ProcessParameterModel).where(ProcessParameterModel.step == step)
            results = session.execute(stmt)
            param_model = results.first()
            if param_model is None:
                params = param_model
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

    def delete_roi(self, sample: Optional[Sample] = None):
        """
        Delete ROIs from a sample.

        Parameters
        ----------
        sample : Sample

        """
        with self.SessionFactory() as session:
            stmt = delete(RoiModel)
            if sample is not None:
                stmt = stmt.where(RoiModel.sample_id == sample.id)
            session.execute(stmt)
            session.commit()

    def add_features(self, roi_list: Sequence[Roi], sample: Sample):
        """
        Stores a list of features in the DB.

        Parameters
        ----------
        roi_list : Sequence[Roi]
        sample : Sample

        """
        feature_model_list = list()
        annotation_model_list = list()
        descriptor_model_list = list()
        descriptor_names = self.feature.descriptor_names()
        for roi in roi_list:
            if roi.features is not None:
                for ft in roi.features:
                    # ft_model = FeatureModel(
                    #     sample_id=sample.id, roi_id=roi.id, data=ft.to_str()
                    # )
                    ft_model = FeatureModel(roi_id=roi.id, data=ft.to_str())
                    ann = ft.annotation
                    annotation_model = AnnotationModel(
                        sample_id=sample.id,
                        roi_id=roi.id,
                        label=ann.label,
                        charge=ann.charge,
                        isotopologue_label=ann.isotopologue_label,
                        isotopologue_index=ann.isotopologue_index,
                    )
                    ft_descriptors = {x: ft.get(x) for x in descriptor_names}

                    descriptor_model = self.DescriptorModel(
                        sample_id=sample.id,
                        roi_id=roi.id,
                        label=ann.label,
                        **ft_descriptors,
                    )

                    feature_model_list.append(ft_model)
                    annotation_model_list.append(annotation_model)
                    descriptor_model_list.append(descriptor_model)
        with self.SessionFactory() as session:
            session.add_all(feature_model_list)
            session.add_all(annotation_model_list)
            session.add_all(descriptor_model_list)
            session.commit()

    def get_features_by_sample(self, sample: Sample) -> list[Feature]:
        """
        Retrieves all features detected in a sample.

        Parameters
        ----------
        sample : Sample

        Returns
        -------
        list[Feature]

        """
        with self.SessionFactory() as session:
            stmt = select(RoiModel).where(RoiModel.sample_id == sample.id)
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

    def get_features_by_label(
        self, label: int, groups: Optional[list[str]] = None
    ) -> list[Feature]:
        """
        Retrieves all samples with an specific feature label.

        Parameters
        ----------
        label : int
            Feature label obtained from feature correspondence.
        groups : list[str] or None, default=None
            Sample group. If provided, only features detected in that group
            are included.

        Returns
        -------
        list[Feature]

        """
        with self.SessionFactory() as session:
            cond = AnnotationModel.label == label
            stmt = select(FeatureModel).join(AnnotationModel)
            if groups is not None:
                cond = cond & (SampleModel.group.in_(groups))
                stmt = stmt.join(SampleModel)
            stmt = stmt.where(cond)

            result = session.execute(stmt)
            feature_list = list()
            for row in result:
                annotation = row.FeatureModel.annotation.to_annotation()
                roi = row.FeatureModel.roi.to_roi(self.roi)
                feature = row.FeatureModel.to_feature(self.feature, roi, annotation)
                feature_list.append(feature)
        return feature_list

    def delete_features(self):
        """
        Delete Features stored in the DB.

        Parameters
        ----------
        sample : Sample

        """
        with self.SessionFactory() as session:
            stmt = delete(FeatureModel)
            session.execute(stmt)
            session.commit()

    def get_descriptors(
        self, sample: Optional[Sample] = None, descriptors: Optional[list[str]] = None
    ):
        """
        Retrieves the descriptors associated with each feature.

        Parameters
        ----------
        sample : Sample or None, default=None
            If provided, retrieves descriptors for features detected in this sample.
            If ``None``, retrieves the descriptors for all samples.

        descriptors : list[str] or None, default=None
            A list of descriptors to retrieve. If ``None``, retrieves all stored
            descriptors.

        Returns
        -------
        dict[str, list[float]]
            A dictionary that maps descriptor names to a list of values for each feature.

        Raises
        ------
        ValueError
            If an invalid descriptor name is provided.

        """
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
        descriptor_dict = {x: list() for x in descriptors}

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
                    descriptor_dict[d].append(row.DescriptorModel.__dict__[d])
        return descriptor_dict


def _check_preprocessing_step(step: Optional[str]):
    if (step is not None) and (step not in c.PREPROCESSING_STEPS):
        valid_steps = ", ".join(c.PREPROCESSING_STEPS)
        msg = f"{step} is not a valid preprocessing step. valid values are {valid_steps}."
        raise ValueError(msg)
