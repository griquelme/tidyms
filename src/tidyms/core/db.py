"""
Storage tools for the Assay class.

AssayData:
    Interface to store/retrieve Sample, ROI, Feature, annotation and
    descriptor data into/from a SQLite Database.

"""

import json
import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy.orm import Mapped, mapped_column, relationship, Session
from sqlalchemy import delete, select, Column, Float, ForeignKey, Integer, String
from sqlalchemy.exc import IntegrityError
from pathlib import Path
from typing import cast, Sequence, Type
from .models import Annotation, Feature, Roi, Sample, SampleData

Base = orm.declarative_base()


class ProcessParameterModel(Base):
    """Stores model parameters, step name and pipeline name."""

    __tablename__ = "processors"
    id: Mapped[str] = mapped_column(sa.String, primary_key=True)
    order: Mapped[int] = mapped_column(Integer)
    parameters: Mapped[str] = mapped_column(sa.String)
    pipeline: Mapped[str] = mapped_column(sa.String)


class SampleModel(Base):
    """Stores data from Sample objects."""

    __tablename__ = "samples"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    path: Mapped[str] = mapped_column(String, nullable=False)
    ms_level: Mapped[int] = mapped_column(Integer)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    group: Mapped[str] = mapped_column(String, nullable=True)
    order: Mapped[int] = mapped_column(Integer, unique=True)
    batch: Mapped[int] = mapped_column(Integer)


class ProcessedSampleModel(Base):
    """Store samples that have been processed by a given pipeline."""

    __tablename__ = "processed_samples"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_id: Mapped[str] = mapped_column(String, ForeignKey("samples.id"))
    processor: Mapped[str] = mapped_column(String, ForeignKey("processors.id"))


class RoiModel(Base):
    """Stores ROI data."""

    __tablename__ = "rois"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sample_id: Mapped[str] = mapped_column(ForeignKey("samples.id"), index=True)
    data: Mapped[str] = mapped_column(String)
    features: Mapped[list["FeatureModel"]] = relationship(back_populates="roi")
    annotations: Mapped[list["AnnotationModel"]] = relationship(back_populates="roi")


class FeatureModel(Base):
    """Store Feature data."""

    __tablename__ = "features"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    roi_id: Mapped[int] = mapped_column(ForeignKey("rois.id"))
    data: Mapped[str] = mapped_column(String)
    roi: Mapped["RoiModel"] = relationship(back_populates="features")
    annotation: Mapped["AnnotationModel"] = relationship(
        back_populates="feature", lazy="immediate"
    )


class AnnotationModel(Base):
    """Store data from Annotation objects."""

    __tablename__ = "annotations"
    id: Mapped[int] = mapped_column(ForeignKey("features.id"), primary_key=True)
    sample_id: Mapped[str] = mapped_column(ForeignKey("samples.id"))
    roi_id: Mapped[int] = mapped_column(ForeignKey("rois.id"))
    label: Mapped[int] = mapped_column(Integer)
    isotopologue_label: Mapped[int] = mapped_column(Integer)
    isotopologue_index: Mapped[int] = mapped_column(Integer)
    charge: Mapped[int] = mapped_column(Integer)
    roi: Mapped["RoiModel"] = relationship(back_populates="annotations")
    feature: Mapped["FeatureModel"] = relationship(back_populates="annotation")


class AssayData:
    """
    Storage class for Assay data.

    Persists assay Samples, ROIs, features and feature descriptors using SQLite.

    Parameters
    ----------
    name : Path or None.
        Path to SQLite DB. If ``None``, uses an in-memory database.
    roi : Type[Roi]
        ROI type of :term:`ROI` stored in the DB.
    feature : Type[Feature]
        Feature type of :term:`feature` stored in the DB.
    echo : bool, default=False
        If ``True``, logs statements emitted to the DB.

    """

    DescriptorModel: Type

    def __init__(
        self,
        path: Path | None,
        roi: Type[Roi],
        feature: Type[Feature],
        echo: bool = False,
    ):
        self.roi = roi
        self.feature = feature

        if path is None:
            db_address = "sqlite:///:memory:"
        else:
            db_address = f"sqlite:///{path}"

        self.engine = sa.create_engine(db_address, echo=echo)
        self.SessionFactory = orm.sessionmaker(bind=self.engine)
        self._create_descriptor_model(feature)

        Base.metadata.create_all(self.engine)

        self._roi_id = 0
        self._feature_id = 0
        self._set_id_values()

    @property
    def roi_id(self):
        """Get the first ROI id available."""
        return self._roi_id

    @property
    def feature_id(self):
        """Get the first feature id available."""
        return self._feature_id

    def _set_id_values(self):
        """Set roi and feature id to index new data."""
        with self.SessionFactory() as session:
            roi_id = session.query(sa.sql.func.max(RoiModel.id)).scalar()
            roi_id = 0 if roi_id is None else roi_id + 1
            self._roi_id = roi_id
            feature_id = session.query(sa.sql.func.max(FeatureModel.id)).scalar()
            feature_id = 0 if feature_id is None else feature_id + 1
            self._feature_id = feature_id

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
                session.add_all([SampleModel(**x.model_dump()) for x in samples])
                session.commit()
            except IntegrityError:
                msg = "Trying to insert a sample with an existing ID"
                raise ValueError(msg)

    def flag_processed(self, samples: list[Sample], pipeline: str):
        """
        Flag samples as processed in a preprocessing step.

        Parameters
        ----------
        samples : list[Sample]
        pipeline : str
            Name of processing pipeline.

        Raises
        ------
        ValueError
            If the preprocessing step name is not a valid name. Valid names
            are defined by `PREPROCESSING_STEPS` in the `_constants` module.

        """
        processed = [
            ProcessedSampleModel(sample_id=x.id, pipeline=pipeline) for x in samples
        ]
        with self.SessionFactory() as session:
            session.add_all(processed)
            session.commit()

    def flag_unprocessed(self, pipeline: str):
        """
        Flag a list of samples as unprocessed.

        Parameters
        ----------
        samples : list[Sample]
        pipeline : str

        """
        # TODO: add samples to unflag
        with self.SessionFactory() as session:
            stmt = delete(ProcessedSampleModel).where(
                ProcessedSampleModel.pipeline == pipeline
            )
            session.execute(stmt)
            session.commit()

    def get_samples(self) -> list[Sample]:
        """
        Retrieve samples stored in the DB.

        Returns
        -------
        list[Sample]

        """
        with self.SessionFactory() as session:
            stmt = select(SampleModel)
            result = session.execute(stmt)
            samples = list()
            for row in result:
                samples.append(_sample_model_to_sample(row.SampleModel))
        return samples

    def get_processed_samples(self, pipeline: str) -> list[Sample]:
        """
        Retrieve processed samples by a pipeline from the DB.

        Parameters
        ----------
        pipeline : str
            Pipeline name

        Returns
        -------
        list[Sample]

        """
        with self.SessionFactory() as session:
            stmt = (
                select(SampleModel)
                .join(ProcessedSampleModel)
                .where(ProcessedSampleModel.pipeline == pipeline)
            )
            processed_samples = list()
            for row in session.execute(stmt):
                sample = _sample_model_to_sample(row.SampleModel)
                processed_samples.append(sample)
        return processed_samples

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

    def set_processing_parameters(self, step: str, pipeline: str, parameters: dict):
        """
        Store preprocessing parameters of a processing step in the DB.

        Parameters
        ----------
        step : str
            Preprocessing step name.
        pipeline : str
            Name of the pipeline that the preprocessing step is part of.
        parameters : dict
            Parameters used in the preprocessing step.

        Raises
        ------
        ValueError
            If the preprocessing step name is not a valid name. Valid names
            are defined by `PREPROCESSING_STEPS` in the `_constants` module.

        """
        param_str = json.dumps(parameters)
        with self.SessionFactory() as session:
            params_model = ProcessParameterModel(
                step=step, parameters=param_str, pipeline=pipeline
            )
            session.add(params_model)
            session.commit()

    def get_processing_parameters(self, step: str) -> dict:
        """
        Retrieve preprocessing parameters of a step from the DB.

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
        with self.SessionFactory() as session:
            stmt = select(ProcessParameterModel).where(ProcessParameterModel.id == step)
            results = session.execute(stmt)
            param_model = results.scalar()
            if param_model is None:
                params = None
            else:
                params = json.loads(param_model.parameters)
        if params is None:
            msg = f"{step} not found in assay data."
            raise ValueError(msg)
        return params

    def get_pipeline_parameters(self, name: str) -> dict[str, dict]:
        """
        Retrieve the parameters of used in each pipeline step.

        Parameters
        ----------
        name : str
            Pipeline name.

        Returns
        -------
        dict[str, dict]
            A dictionary where each key is the name of each step and values
            are the processing parameters.

        """
        with self.SessionFactory() as session:
            stmt = select(ProcessParameterModel).where(
                ProcessParameterModel.pipeline == name
            )
            parameters = dict()
            for row in session.execute(stmt):
                param_model = row.ProcessParameterModel
                params = json.loads(param_model.parameters)
                parameters[param_model.step] = params
        return parameters

    def add_pipeline_parameters(self, name: str, parameters: dict[str, dict]):
        """
        Store parameters used in a pipeline.

        Parameters
        ----------
        name : str
            Pipeline name.
        parameters : dict[str, dict]
            Pipeline parameters.
        """
        parameter_model_list = list()
        for step, param in parameters.items():
            param_str = json.dumps(param)
            model = ProcessParameterModel(
                step=step, parameters=param_str, pipeline=name
            )
            parameter_model_list.append(model)

        with self.SessionFactory() as session:
            session.add_all(parameter_model_list)
            session.commit()

    def update_pipeline_parameters(self, name: str, parameters: dict[str, dict]):
        """
        Update parameters of a pipeline.

        Parameters
        ----------
        name : str
            Pipeline name
        parameters : dict[str, dict]
            New parameters.

        """
        update = list()
        for step, param in parameters.items():
            update.append(
                {"pipeline": name, "step": step, "parameters": json.dumps(param)}
            )

        with self.SessionFactory() as session:
            session.execute(sa.update(ProcessParameterModel), update)
            session.commit()

    def _label_roi_list(self, *rois: Roi):
        """Add a id label to ROIs."""
        for k_roi, roi in enumerate(rois, start=self._roi_id):
            roi.id = k_roi
        self._roi_id += len(rois)

    def add_roi_list(self, sample: Sample, *rois: Roi):
        """
        Store a list of ROI extracted from a sample.

        Parameters
        ----------
        sample : Sample
        *rois : Roi

        """
        roi_model_list = list()
        self._label_roi_list(*rois)
        for roi in rois:
            roi_model = RoiModel(sample_id=sample.id, data=roi.to_str(), id=roi.id)
            roi_model_list.append(roi_model)

        with self.SessionFactory() as session:
            session.add_all(roi_model_list)
            session.commit()

    def get_roi_list(self, sample: Sample, load_features: bool = True) -> list[Roi]:
        """
        Retrieve a list of ROI detected in a sample.

        Parameters
        ----------
        sample : Sample
        bool : Wether to load or not feature data detected in the ROI.

        Returns
        -------
        list[Roi]

        """
        with self.SessionFactory() as session:
            if load_features:
                stmt = (
                    select(RoiModel)
                    .where(RoiModel.sample_id == sample.id)
                    .options(orm.immediateload(RoiModel.features))
                )
            else:
                stmt = select(RoiModel).where(RoiModel.sample_id == sample.id)
            result = session.execute(stmt)
            roi_list = list()
            for row in result:
                model = row.RoiModel
                roi = _roi_model_to_roi(
                    model, self.roi, self.feature, load_features=load_features
                )
                roi_list.append(roi)
        return roi_list

    def get_roi_by_id(self, roi_id: int, load_features: bool = True) -> Roi:
        """
        Retrieve a ROI by id.

        Parameters
        ----------
        roi_id : str

        Returns
        -------
        Roi

        """
        with self.SessionFactory() as session:
            if load_features:
                stmt = (
                    select(RoiModel)
                    .where(RoiModel.id == roi_id)
                    .options(orm.immediateload(RoiModel.features))
                )
            else:
                stmt = select(RoiModel).where(RoiModel.id == roi_id)
            result = session.execute(stmt).scalar()
            if result is not None:
                roi = _roi_model_to_roi(
                    result, self.roi, self.feature, load_features=load_features
                )
            else:
                roi = None

        if roi is None:
            msg = f"No ROI stored with id={roi_id}."
            raise ValueError(msg)
        return roi

    def delete_roi(self, sample: Sample | None = None):
        """
        Delete ROIs from a sample.

        Parameters
        ----------
        sample : Sample

        """
        # TODO: cascade and delete features
        with self.SessionFactory() as session:
            stmt = delete(RoiModel)
            if sample is not None:
                stmt = stmt.where(RoiModel.sample_id == sample.id)
            session.execute(stmt)
            session.commit()

    def _label_feature_list(self, feature_list: Sequence[Feature]):
        """Add an id label for each feature."""
        for k, ft in enumerate(feature_list, start=self._feature_id):
            ft.id = k
        self._feature_id += len(feature_list)

    def _add_feature_data(self, feature_list: Sequence[Feature], session: Session):
        """Add feature data to a Session."""
        feature_model_list = list()
        for ft in feature_list:
            model = FeatureModel(roi_id=ft.roi.id, data=ft.to_str(), id=ft.id)
            feature_model_list.append(model)
        session.add_all(feature_model_list)

    def _add_feature_annotation_data(
        self, feature_list: Sequence[Feature], sample: Sample, session: Session
    ):
        """Add feature annotation data to session."""
        feature_annotation_model_list = list()
        for ft in feature_list:
            model = AnnotationModel(
                id=ft.id,
                sample_id=sample.id,
                roi_id=ft.roi.id,
                label=ft.annotation.label,
                charge=ft.annotation.charge,
                isotopologue_label=ft.annotation.isotopologue_label,
                isotopologue_index=ft.annotation.isotopologue_index,
            )
            feature_annotation_model_list.append(model)
        session.add_all(feature_annotation_model_list)

    def _add_feature_descriptors_data(
        self, feature_list: Sequence[Feature], sample: Sample, session: Session
    ):
        """Add descriptor information to session."""
        feature_annotation_model_list = list()
        for ft in feature_list:
            ft_descriptors = ft.describe()
            model = self.DescriptorModel(
                id=ft.id,
                sample_id=sample.id,
                roi_id=ft.roi.id,
                label=ft.annotation.label,
                **ft_descriptors,
            )
            feature_annotation_model_list.append(model)
        session.add_all(feature_annotation_model_list)

    def add_features(self, sample: Sample, *features: Feature):
        """
        Store a list of features in the DB.

        Parameters
        ----------
        sample : Sample
        *features: Feature

        """
        self._label_feature_list(features)
        with self.SessionFactory() as session:
            self._add_feature_data(features, session)
            self._add_feature_annotation_data(features, sample, session)
            self._add_feature_descriptors_data(features, sample, session)
            session.commit()

    def get_features_by_sample(self, sample: Sample) -> list[Feature]:
        """
        Retrieve all features detected in a sample.

        Parameters
        ----------
        sample : Sample

        Returns
        -------
        list[Feature]

        """
        with self.SessionFactory() as session:
            stmt = (
                select(RoiModel)
                .where(RoiModel.sample_id == sample.id)
                .options(orm.immediateload(RoiModel.features))
            )
            result = session.execute(stmt)
            feature_list = list()
            for row in result:
                rm = row.RoiModel
                roi = _roi_model_to_roi(rm, self.roi, self.feature, load_features=True)
                for ft in roi.features:
                    feature_list.append(ft)
        return feature_list

    def get_features_by_label(
        self, label: int, groups: list[str] | None = None
    ) -> list[Feature]:
        """
        Retrieve all samples with an specific feature label.

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
            stmt = (
                select(FeatureModel)
                .join(AnnotationModel)
                .options(orm.immediateload(FeatureModel.roi))
            )
            if groups is not None:
                cond = cond & (SampleModel.group.in_(groups))
                stmt = stmt.join(SampleModel)
            stmt = stmt.where(cond)

            result = session.execute(stmt)
            feature_list = list()
            for row in result:
                roi = _roi_model_to_roi(
                    row.FeatureModel.roi, self.roi, self.feature, load_features=False
                )
                feature = _feature_model_to_feature(row.FeatureModel, self.feature, roi)
                feature_list.append(feature)
        return feature_list

    def get_features_by_id(self, ft_id: int) -> Feature:
        """
        Retrieve all samples with an specific feature label.

        Parameters
        ----------
        ft_id : int

        Returns
        -------
        Feature

        """
        with self.SessionFactory() as session:
            stmt = (
                select(FeatureModel)
                .join(AnnotationModel)
                .where(FeatureModel.id == ft_id)
                .options(orm.immediateload(FeatureModel.roi))
            )
            result = session.execute(stmt).scalar()
            if result is not None:
                roi = _roi_model_to_roi(
                    result.roi, self.roi, self.feature, load_features=False
                )
                feature = _feature_model_to_feature(result, self.feature, roi)

            else:
                feature = None

        if feature is None:
            msg = f"No feature stored found with id={ft_id}"
            raise ValueError(msg)

        return feature

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
        self, sample: Sample | None = None, descriptors: list[str] | None = None
    ):
        """
        Retrieve the descriptors associated with each feature.

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
        dict[str, list]
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

        descriptors = descriptors + ["id", "roi_id", "sample_id", "label"]
        descriptor_dict = {x: list() for x in descriptors}

        with self.SessionFactory() as session:
            if sample is None:
                stmt = select(self.DescriptorModel)
            else:
                stmt = select(self.DescriptorModel).where(
                    self.DescriptorModel.sample_id == sample.id  # type: ignore
                )
            result = session.execute(stmt)
            for row in result:
                for d in descriptors:
                    descriptor_dict[d].append(row.DescriptorModel.__dict__[d])
        return descriptor_dict

    def get_annotations(self) -> dict[str, list[int]]:
        """Get annotations of features."""
        columns = [
            "id",
            "label",
            "sample_id",
            "isotopologue_index",
            "isotopologue_label",
            "charge",
        ]
        annotation_dict: dict[str, list[int]] = {x: list() for x in columns}
        with self.SessionFactory() as session:
            stmt = select(AnnotationModel).where(AnnotationModel.label != -1)
            result = session.execute(stmt)
            for row in result:
                for c in columns:
                    val = cast(int, row.AnnotationModel.__dict__[c])
                    annotation_dict[c].append(val)
        return annotation_dict

    def get_sample_data(self, sample_id: str) -> SampleData:
        """Retrieve a SampleData instance."""
        sample = self.search_sample(sample_id)
        # roi_list = self.get_roi_list(sample)
        return SampleData(sample)

    def search_sample(self, sample_id: str) -> Sample:
        """
        Search a sample in the DB.

        Parameters
        ----------
        sample_id : str

        Returns
        -------
        Sample

        Raises
        ------
        ValueError
            If no sample with the specified id exists.

        """
        with self.SessionFactory() as session:
            stmt = select(SampleModel).where(SampleModel.id == sample_id)
            result = session.execute(stmt).scalar()
        if result is None:
            msg = f"No sample with id={sample_id} was found."
            raise ValueError(msg)
        return _sample_model_to_sample(result)

    def store_sample_data(self, data: SampleData):
        """Store a SampleData instance."""
        self.add_roi_list(data.sample, *data.get_roi_list())
        self.add_features(data.sample, *data.get_feature_list())

    def update_feature_labels(self, labels: dict[int, int]):
        """
        Update group label of features.

        No check is done on the correctness of the annotation. The user must
        check that a unique feature label per sample was assigned.

        Parameters
        ----------
        labels : dict[int, int]
            A mapping from feature ids to group labels.

        """
        update = [{"id": k, "label": v} for k, v in labels.items()]
        with self.SessionFactory() as session:
            session.execute(sa.update(AnnotationModel), update)
            session.execute(sa.update(self.DescriptorModel), update)
            session.commit()


def _create_descriptor_table(feature_type: Type[Feature], metadata: sa.MetaData):
    """
    Create a Model for descriptors of features.

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
        Column("id", Integer, ForeignKey("features.id"), primary_key=True),
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


def _roi_model_to_roi(
    model: RoiModel,
    roi_type: Type[Roi],
    feature_type: Type[Feature],
    load_features: bool = False,
) -> Roi:
    """Create a ROI instance."""
    roi = roi_type.from_str(model.data)
    roi.id = model.id
    if load_features:
        for ft_model in model.features:
            ft = _feature_model_to_feature(ft_model, feature_type, roi)
            roi.add_feature(ft)
    return roi


def _feature_model_to_feature(
    model: FeatureModel, feature_type: Type[Feature], roi: Roi
) -> Feature:
    """Create a Feature instance from a FeatureModel instance."""
    ann_model = model.annotation
    annotation = _annotation_model_to_annotation(ann_model)
    return feature_type.from_str(model.data, roi, annotation)


def _annotation_model_to_annotation(model: AnnotationModel) -> Annotation:
    """Create an Annotation instance from an AnnotationModel instance."""
    return Annotation(
        label=model.label,
        isotopologue_label=model.isotopologue_label,
        isotopologue_index=model.isotopologue_index,
        charge=model.charge,
    )


def _sample_model_to_sample(model: SampleModel) -> Sample:
    """Convert to a Sample object."""
    return Sample(
        path=Path(model.path),
        id=model.id,
        ms_level=model.ms_level,
        start_time=model.start_time,
        end_time=model.end_time,
        group=model.group,
        order=model.order,
        batch=model.batch,
    )
