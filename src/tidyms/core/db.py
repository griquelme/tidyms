"""
Storage tools for the Assay class.

AssayData:
    Interface to store/retrieve Sample, ROI, Feature, annotation and
    descriptor data into/from a SQLite Database.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generic, Sequence

import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy import Column, Float, ForeignKey, Integer, String, delete, select
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from . import exceptions
from .constants import IDFeatureDescriptors
from .models import Annotation, Feature, FeatureGroupMap, FeatureType, Roi, RoiType, Sample
from .processors import ProcessingPipeline, ProcessorConfiguration, ProcessorRegistry, ProcessStatus, SampleData

Base = orm.declarative_base()


class ProcessorModel(Base):
    """Stores model parameters, step name and pipeline name."""

    __tablename__ = "processors"
    id: Mapped[str] = mapped_column(sa.String, primary_key=True)
    order: Mapped[int] = mapped_column(Integer)
    parameters: Mapped[str] = mapped_column(sa.String)
    type: Mapped[str] = mapped_column(sa.String)
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
    extra: Mapped[str] = mapped_column(String, nullable=True)
    reader: Mapped[str] = mapped_column(String, nullable=True)


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
    sample_id: Mapped[str] = mapped_column(ForeignKey("samples.id"), index=True)
    data: Mapped[str] = mapped_column(String)
    roi: Mapped["RoiModel"] = relationship(back_populates="features")
    annotation: Mapped["AnnotationModel"] = relationship(back_populates="feature", lazy="immediate")


class AnnotationModel(Base):
    """Store data from Annotation objects."""

    __tablename__ = "annotations"
    id: Mapped[int] = mapped_column(ForeignKey("features.id"), primary_key=True)
    sample_id: Mapped[str] = mapped_column(ForeignKey("samples.id"))
    roi_id: Mapped[int] = mapped_column(ForeignKey("rois.id"))
    group: Mapped[int] = mapped_column(Integer)
    isotopologue_label: Mapped[int] = mapped_column(Integer)
    isotopologue_index: Mapped[int] = mapped_column(Integer)
    charge: Mapped[int] = mapped_column(Integer)
    roi: Mapped["RoiModel"] = relationship(back_populates="annotations")
    feature: Mapped["FeatureModel"] = relationship(back_populates="annotation", lazy="immediate")


class AnnotationPatchModel(Base):
    """Store patched annotation data."""

    __tablename__ = "annotation_patches"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    field: Mapped[str] = mapped_column(String)
    value: Mapped[int] = mapped_column(Integer)


class DescriptorPatchModel(Base):
    """Store patched descriptor data."""

    __tablename__ = "descriptor_patches"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    descriptor: Mapped[str] = mapped_column(String)
    value: Mapped[float] = mapped_column(Float)


class MissingDataModel(Base):
    """Store data matrix inputted missing data."""

    __tablename__ = "missing_data"
    sample_id: Mapped[str] = mapped_column(String, primary_key=True)
    group: Mapped[int] = mapped_column(Integer, primary_key=True)
    value: Mapped[float] = mapped_column(Float)


class AssayData(Generic[FeatureType, RoiType]):
    """
    Storage class for Assay data.

    Persists assay Samples, ROIs, features and feature descriptors using SQLite.

    Parameters
    ----------
    name : Path or None.
        Path to SQLite DB. If ``None``, uses an in-memory database.
    roi : type[Roi]
        ROI type of :term:`ROI` stored in the DB.
    feature : Type[Feature]
        Feature type of :term:`feature` stored in the DB.
    echo : bool, default=False
        If ``True``, logs statements emitted to the DB.

    """

    DescriptorModel: type

    def __init__(
        self,
        path: Path | None,
        roi: type[RoiType],
        feature: type[FeatureType],
        echo: bool = False,
    ):
        self.roi = roi
        self.feature = feature
        self.status = ProcessStatus()

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

    def add_sample_data(
        self,
        data: SampleData[FeatureType, RoiType],
        descriptors: list[dict] | None = None,
        annotations: list[dict] | None = None,
    ) -> None:
        """Store a SampleData instance."""
        if data.sample.id in self.list_samples():
            raise exceptions.SampleIdAlreadyInAssay(data.sample.id)

        self._label_sample_data(data)
        features = data.get_features()

        if descriptors is None:
            descriptors = data.get_feature_descriptors()
            self._fix_descriptor_labels(features, descriptors)

        if annotations is None:
            annotations = data.get_feature_annotations()
            self._fix_annotation_labels(features, annotations)

        with self.SessionFactory() as session:
            self._add_samples(session, data.sample)
            self._add_roi_list(data.sample, session, *data.roi)
            self._add_features(session, data.sample.id, descriptors, annotations, *features)
            session.commit()

    def fetch_sample_data(self, sample_id: str) -> SampleData[FeatureType, RoiType]:
        """Store a SampleData instance."""
        sample = self._fetch_sample(sample_id)
        rois = self.get_roi_list(sample_id, load_features=True)
        return SampleData(sample=sample, roi=rois)

    def delete_sample_data(self, sample_id: str) -> None:
        """Delete a sample data."""
        with self.SessionFactory() as session:
            stmt = delete(SampleModel).where(SampleModel.id == sample_id)
            session.execute(stmt)

            stmt = delete(RoiModel).where(RoiModel.sample_id == sample_id)
            session.execute(stmt)

            stmt = delete(FeatureModel).where(FeatureModel.sample_id == sample_id)
            session.execute(stmt)

            session.commit()

    def add_pipeline(self, pipeline: ProcessingPipeline):
        """
        Store pipeline processing parameters.

        Parameters
        ----------
        pipeline : ProcessingPipeline

        """
        # TODO: check multiple pipelines with same id
        processors = [ProcessorModel(**x.to_config().model_dump()) for x in pipeline.processors]
        with self.SessionFactory() as session:
            session.add_all(processors)
            session.commit()

    def fetch_pipeline(self, pipeline_id: str) -> ProcessingPipeline:
        """
        Retrieve a processing pipeline.

        Parameters
        ----------
        pipeline_id : str

        """
        with self.SessionFactory() as session:
            stmt = select(ProcessorModel).where(ProcessorModel.pipeline == pipeline_id)
            results = session.execute(stmt)
            config_list = list()
            for row in results:
                proc = row.ProcessorModel
                proc_info = ProcessorConfiguration(
                    id=proc.id,
                    pipeline=proc.pipeline,
                    order=proc.order,
                    parameters=proc.parameters,
                    type=proc.type,
                )
                config_list.append(proc_info)
            config_list = sorted(config_list, key=lambda x: x.order)
        if not config_list:
            raise exceptions.PipelineNotFound(pipeline_id)
        processors = [ProcessorRegistry.create_from_config(x) for x in config_list]
        return ProcessingPipeline(id=pipeline_id, processors=processors)

    def delete_pipeline(self, pipeline_id: str) -> None:
        """
        Delete a pipeline from the DB.

        Parameters
        ----------
        pipeline_id : str
            The pipeline id to delete.

        """
        with self.SessionFactory() as session:
            stmt = delete(ProcessorModel).where(ProcessorModel.pipeline == pipeline_id)
            session.execute(stmt)
            session.commit()

    def fetch_descriptors(self, sample_id: str | None = None, descriptors: Sequence[str] | None = None) -> dict:
        """
        Retrieve the feature descriptors.

        Parameters
        ----------
        sample_id : str or None, default=None
            If provided, retrieves descriptors from specified sample. If ``None``,
            retrieves descriptors from features in all samples.
        descriptors : list[str] or None, default=None
            A list of descriptors to retrieve. If ``None``, retrieves all stored
            descriptors.

        Returns
        -------
        dict
            A dictionary that maps descriptor names to a list of values for each
            feature. Besides descriptors, `id` provides the feature id
            associated with each descriptor, `roi_id` is the ROI id where
            the feature was extracted from and `sample_id` is the sample id
            where the ROI was extracted from.

        Raises
        ------
        ValueError
            If an invalid descriptor name is provided.

        """
        valid_descriptors = self.feature.descriptor_names()

        if descriptors is None:
            requested_descriptors = list(valid_descriptors)
        else:
            requested_descriptors = list(descriptors)

        for d in requested_descriptors:
            if d not in valid_descriptors:
                msg = f"{d} is not a valid descriptor of feature type {self.feature.__name__}."
                raise ValueError(msg)

        requested_descriptors.extend([x.value for x in IDFeatureDescriptors])
        descriptor_dict = {x: list() for x in requested_descriptors}

        with self.SessionFactory() as session:
            stmt = select(self.DescriptorModel)
            if sample_id is not None:
                stmt = stmt.where(self.DescriptorModel.sample_id == sample_id)  # type: ignore
            result = session.execute(stmt)
            for row in result:
                for d in requested_descriptors:
                    descriptor_dict[d].append(row.DescriptorModel.__dict__[d])
        return descriptor_dict

    def fetch_annotations(self, sample_id: str | None = None, has_feature_group: bool = False) -> list[Annotation]:
        """Get annotations of features."""
        stmt = select(AnnotationModel)
        if has_feature_group:
            where_clause = AnnotationModel.group != -1
            if sample_id is not None:
                where_clause = where_clause & (AnnotationModel.sample_id == sample_id)
            stmt = stmt.where(where_clause)
        elif sample_id is not None:
            stmt = stmt.where(AnnotationModel.sample_id == sample_id)

        with self.SessionFactory() as session:
            result = session.execute(stmt)
            annotations = list()
            for row in result:
                ann = Annotation(**row.AnnotationModel.__dict__)
                annotations.append(ann)
        return annotations

    def set_feature_groups(self, groups: list[FeatureGroupMap]):
        """
        Set feature group of features.

        No check is done on the correctness of the annotation. The user must
        check that a unique feature label per sample was assigned.

        Parameters
        ----------
        labels : dict[int, int]
            A mapping from feature ids to group labels.

        """
        with self.SessionFactory() as session:
            session.execute(sa.update(AnnotationModel), groups)
            session.execute(sa.update(self.DescriptorModel), groups)
            session.commit()

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
    def _create_descriptor_model(cls, feature_type: type[FeatureType]):
        """
        Create the DescriptorModel table using descriptors available in the Feature class.

        Parameters
        ----------
        feature_type : Feature
            Feature type stored in the AssayData.

        """
        if not hasattr(cls, "DescriptorModel"):
            cls.DescriptorModel = _create_descriptor_table(feature_type, Base.metadata)

    def _add_samples(self, session: Session, *samples: Sample) -> None:
        """
        Add samples into the DB.

        Parameters
        ----------
        samples : Sample

        Raises
        ------
        ValueError
            If a sample with the same id already exists in the DB.

        """
        try:
            session.add_all([SampleModel(**x.model_dump()) for x in samples])
        except IntegrityError as e:
            msg = "Trying to insert a sample with an existing ID"
            raise exceptions.SampleIdAlreadyInAssay(msg) from e

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
        processed = [ProcessedSampleModel(sample_id=x.id, pipeline=pipeline) for x in samples]
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
            stmt = delete(ProcessedSampleModel).where(ProcessedSampleModel.pipeline == pipeline)
            session.execute(stmt)
            session.commit()

    def list_samples(self) -> list[str]:
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
                samples.append(row.SampleModel.id)
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
            stmt = select(SampleModel).join(ProcessedSampleModel).where(ProcessedSampleModel.pipeline == pipeline)
            processed_samples = list()
            for row in session.execute(stmt):
                sample = _sample_model_to_sample(row.SampleModel)
                processed_samples.append(sample)
        return processed_samples

    def _delete_sample(self, sample_id: str) -> None:
        """
        Delete samples stored in the DB.

        Parameters
        ----------
        sample_ids : list[str]
            Unique IDs of samples in the DB.

        """
        with self.SessionFactory() as session:
            stmt = delete(SampleModel).where(SampleModel.id == sample_id)
            session.execute(stmt)
            session.commit()

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
            stmt = select(ProcessorModel).where(ProcessorModel.pipeline == name)
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
            model = ProcessorModel(step=step, parameters=param_str, pipeline=name)
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
            update.append({"pipeline": name, "step": step, "parameters": json.dumps(param)})

        with self.SessionFactory() as session:
            session.execute(sa.update(ProcessorModel), update)
            session.commit()

    def _fetch_sample(self, sample_id: str) -> Sample:
        """Fetch a sample from the db."""
        with self.SessionFactory() as session:
            stmt = select(SampleModel).where(SampleModel.id == sample_id)
            try:
                model = session.scalars(stmt).one()
                return _sample_model_to_sample(model)
            except NoResultFound as e:
                raise exceptions.SampleDataNotFound(sample_id) from e

    def _add_roi_list(self, sample: Sample, session: Session, *rois: RoiType):
        """
        Store a list of ROI extracted from a sample.

        Parameters
        ----------
        sample : Sample
        *rois : Roi

        """
        roi_model_list = list()
        for roi in rois:
            roi_model = RoiModel(sample_id=sample.id, data=roi.to_str(), id=roi.id)
            roi_model_list.append(roi_model)

        session.add_all(roi_model_list)

    def get_roi_list(self, sample_id: str, load_features: bool = True) -> list[RoiType]:
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
                    .where(RoiModel.sample_id == sample_id)
                    .options(orm.immediateload(RoiModel.features))
                )
            else:
                stmt = select(RoiModel).where(RoiModel.sample_id == sample_id)
            result = session.execute(stmt)
            roi_list = list()
            for row in result:
                model = row.RoiModel
                roi = _roi_model_to_roi(model, self.roi, self.feature, load_features=load_features)
                roi_list.append(roi)
        return roi_list

    def get_roi_by_id(self, roi_id: int, load_features: bool = True) -> RoiType:
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
                stmt = select(RoiModel).where(RoiModel.id == roi_id).options(orm.immediateload(RoiModel.features))
            else:
                stmt = select(RoiModel).where(RoiModel.id == roi_id)
            result = session.execute(stmt).scalar()
            if result is not None:
                roi = _roi_model_to_roi(result, self.roi, self.feature, load_features=load_features)
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

    def _label_sample_data(self, data: SampleData) -> None:
        """Label ROIs and features in sample data."""
        for roi in data.roi:
            roi.id = self._roi_id
            self._roi_id += 1
            for ft in roi.features:
                ft.annotation.id = self._feature_id
                ft.annotation.roi_id = ft.roi.id
                ft.annotation.sample_id = data.sample.id
                self._feature_id += 1

    def _fix_annotation_labels(self, features: list[FeatureType], annotations: list[dict]):
        """Add annotation labels provided by DB."""
        for ft, ann in zip(features, annotations):
            ann[IDFeatureDescriptors.ROI_ID] = ft.roi.id
            ann[IDFeatureDescriptors.ID] = ft.annotation.id

    def _fix_descriptor_labels(self, features: list[FeatureType], descriptors: list[dict]):
        """Add annotation labels provided by DB."""
        for ft, d in zip(features, descriptors):
            d[IDFeatureDescriptors.ROI_ID] = ft.roi.id
            d[IDFeatureDescriptors.ID] = ft.annotation.id

    def _label_feature_list(self, feature_list: Sequence[FeatureType]):
        """Add an id label for each feature."""
        for k, ft in enumerate(feature_list, start=self._feature_id):
            ft.annotation.id = k
        self._feature_id += len(feature_list)

    def _add_feature_data(self, feature_list: Sequence[FeatureType], session: Session, sample_id: str):
        """Add feature data to a Session."""
        feature_model_list = list()
        for ft in feature_list:
            model = FeatureModel(roi_id=ft.roi.id, data=ft.to_str(), id=ft.annotation.id, sample_id=sample_id)
            feature_model_list.append(model)
        session.add_all(feature_model_list)

    def _add_feature_annotation_data(self, annotations: list[dict], session: Session):
        """Add feature annotation data to session."""
        feature_annotation_model_list = list()
        for annotation in annotations:
            model = AnnotationModel(**annotation)
            feature_annotation_model_list.append(model)
        session.add_all(feature_annotation_model_list)

    def _add_feature_descriptors_data(self, descriptors: list[dict], session: Session):
        """Add descriptor information to session."""
        feature_annotation_model_list = list()
        for descriptor in descriptors:
            model = self.DescriptorModel(**descriptor)
            feature_annotation_model_list.append(model)
        session.add_all(feature_annotation_model_list)

    def _add_features(
        self,
        session: Session,
        sample_id: str,
        descriptors: list[dict],
        annotations: list[dict],
        *features: FeatureType,
    ):
        """
        Store a list of features in the DB.

        Parameters
        ----------
        sample : Sample
        *features: Feature

        """
        self._add_feature_data(features, session, sample_id)
        self._add_feature_annotation_data(annotations, session)
        self._add_feature_descriptors_data(descriptors, session)

    def get_features_by_sample(self, sample_id: str) -> list[FeatureType]:
        """
        Retrieve all features detected in a sample.

        Parameters
        ----------
        sample : Sample

        Returns
        -------
        list[Feature]

        """
        sample_data = self.fetch_sample_data(sample_id)
        return sample_data.get_features()

    def get_features_by_label(self, label: int, groups: list[str] | None = None) -> list[FeatureType]:
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
            stmt = select(FeatureModel).join(AnnotationModel).options(orm.immediateload(FeatureModel.roi))
            if groups is not None:
                cond = cond & (SampleModel.group.in_(groups))
                stmt = stmt.join(SampleModel)
            stmt = stmt.where(cond)

            result = session.execute(stmt)
            feature_list = list()
            for row in result:
                roi = _roi_model_to_roi(row.FeatureModel.roi, self.roi, self.feature, load_features=False)
                feature = _feature_model_to_feature(row.FeatureModel, self.feature, roi)
                feature_list.append(feature)
        return feature_list

    def get_features_by_id(self, ft_id: int) -> FeatureType:
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
                roi = _roi_model_to_roi(result.roi, self.roi, self.feature, load_features=False)
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


def _create_descriptor_table(feature_type: type[Feature], metadata: sa.MetaData):
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
        Column(IDFeatureDescriptors.ID.value, Integer, ForeignKey("features.id"), primary_key=True),
        Column(IDFeatureDescriptors.SAMPLE_ID.value, String, ForeignKey("samples.id")),
        Column(IDFeatureDescriptors.ROI_ID.value, Integer, ForeignKey("rois.id")),
        Column(IDFeatureDescriptors.FEATURE_GROUP.value, Integer, ForeignKey("annotations.group")),
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
    roi_type: type[RoiType],
    feature_type: type[FeatureType],
    load_features: bool = False,
) -> RoiType:
    """Create a ROI instance."""
    roi = roi_type.from_str(model.data)
    if load_features:
        for ft_model in model.features:
            ft = _feature_model_to_feature(ft_model, feature_type, roi)
            roi.add_feature(ft)
    return roi


def _feature_model_to_feature(model: FeatureModel, feature_type: type[FeatureType], roi: Roi) -> FeatureType:
    """Create a Feature instance from a FeatureModel instance."""
    annotation = Annotation(**model.annotation.__dict__)
    return feature_type.from_str(model.data, roi, annotation)


def _sample_model_to_sample(model: SampleModel) -> Sample:
    """Convert to a Sample object."""
    model_dict = {field.name: getattr(model, field.name) for field in model.__table__.c}
    return Sample(**model_dict)
