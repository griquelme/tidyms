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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence, Type, Union


# TODO: test all code
# TODO: Remove sample data.
# TODO: make add_roi and similar functions private.
# TODO: add update_feature_label
# TODO: add remove empty ROI
# TODO: add remove non-matched.


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


class SampleData:
    """
    Container class for the associated with a sample.

    Attributes
    ----------
    sample : Sample
    roi : Optional[Sequence[Roi]]

    """

    def __init__(self, sample: Sample, roi: Optional[Sequence[Roi]] = None) -> None:
        self.sample = sample
        if roi is None:
            roi = list()
        self.roi = roi

    def get_feature_list(self) -> Sequence[Feature]:
        feature_list = list()
        for roi in self.roi:
            if roi.features is not None:
                feature_list.extend(roi.features)
        return feature_list


Base = orm.declarative_base()


class ProcessParameterModel(Base):
    """
    Model for the Table with processing parameters of the Assay.

    """

    __tablename__ = "parameters"
    step: Mapped[str] = mapped_column(sa.String, primary_key=True)
    parameters: Mapped[str] = mapped_column(sa.String)
    pipeline: Mapped[str] = mapped_column(sa.String)


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
    pipeline: Mapped[str] = mapped_column(String, ForeignKey("parameters.pipeline"))


class RoiModel(Base):
    """
    Model for ROI extracted in each Sample.

    """

    __tablename__ = "rois"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
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
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
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
    id: Mapped[int] = mapped_column(ForeignKey("features.id"), primary_key=True)
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

        # set values for ROI id and Feature id
        with self.SessionFactory() as session:
            roi_id = session.query(sa.sql.func.max(RoiModel.id)).scalar()
            if roi_id is None:
                roi_id = 0
            else:
                roi_id += 1
            self._roi_id = roi_id
            feature_id = session.query(sa.sql.func.max(FeatureModel.id)).scalar()
            if feature_id is None:
                feature_id = 0
            else:
                feature_id += 1
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
                session.add_all([SampleModel(**x.to_dict()) for x in samples])
                session.commit()
            except IntegrityError:
                msg = "Trying to insert a sample with an existing ID"
                raise ValueError(msg)

    def flag_processed(self, samples: list[Sample], pipeline: str):
        """
        Flags samples as processed in a preprocessing step.

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
        Flags a list of samples as unprocessed.

        Parameters
        ----------
        samples : list[Sample]
        pipeline : str

        """
        with self.SessionFactory() as session:
            stmt = delete(ProcessedSampleModel).where(
                ProcessedSampleModel.pipeline == pipeline
            )
            session.execute(stmt)
            session.commit()

    def get_samples(self) -> list[Sample]:
        """
        Retrieves samples stored in the DB.

        Returns
        -------
        list[Sample]

        """
        with self.SessionFactory() as session:
            stmt = select(SampleModel)
            result = session.execute(stmt)
            samples = list()
            for row in result:
                samples.append(row.SampleModel.to_sample())
        return samples

    def get_processed_samples(self, pipeline: str) -> list[Sample]:
        with self.SessionFactory() as session:
            stmt = (
                select(SampleModel)
                .join(ProcessedSampleModel)
                .where(ProcessedSampleModel.pipeline == pipeline)
            )
            processed_samples = list()
            for row in session.execute(stmt):
                sample = row.SampleModel.to_sample()
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
        Stores preprocessing parameters of a step in the DB.

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
        with self.SessionFactory() as session:
            stmt = select(ProcessParameterModel).where(ProcessParameterModel.step == step)
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
        with self.SessionFactory() as session:
            stmt = select(ProcessParameterModel).where(ProcessParameterModel.pipeline == name)
            parameters = dict()
            for row in session.execute(stmt):
                param_model = row.ProcessParameterModel
                params = json.loads(param_model.parameters)
                parameters[param_model.step] = params
        return parameters

    def add_pipeline_parameters(self, name: str, parameters: dict[str, dict]):
        parameter_model_list = list()
        for step, param in parameters.items():
            param_str = json.dumps(param)
            model = ProcessParameterModel(step=step, parameters=param_str, pipeline=name)
            parameter_model_list.append(model)

        with self.SessionFactory() as session:
            session.add_all(parameter_model_list)
            session.commit()

    def update_pipeline_parameters(self, name: str, parameters: dict[str, dict]):
        update = list()
        for step, param in parameters.items():
            update.append({"pipeline": name, "step": step, "parameters": json.dumps(param)})

        with self.SessionFactory() as session:
            session.execute(sa.update(ProcessParameterModel), update)
            session.commit()

    def add_roi_list(self, roi_list: Sequence[Roi], sample: Sample):
        """
        Stores a list of ROI extracted from a sample in the DB.

        Parameters
        ----------
        roi_list : Sequence[Roi]
        sample : Sample

        """
        roi_model_list = list()
        for k, roi in enumerate(roi_list, start=self._roi_id):
            roi.id = k
            roi_model = RoiModel(sample_id=sample.id, data=roi.to_string(), id=roi.id)
            roi_model_list.append(roi_model)

        with self.SessionFactory() as session:
            session.add_all(roi_model_list)
            session.commit()
        self._roi_id += len(roi_list)

    def get_roi_list(self, sample: Sample) -> Sequence[Roi]:
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

    def get_roi_by_id(self, roi_id: int) -> Roi:
        """
        Retrieves a ROI by id

        Parameters
        ----------
        roi_id : str

        Returns
        -------
        Roi

        """
        with self.SessionFactory() as session:
            stmt = select(RoiModel).where(RoiModel.id == roi_id)
            result = session.execute(stmt).scalar()
            if result is not None:
                roi = result.to_roi(self.roi)
                features = list()
                for ft_model in result.features:
                    annotation = ft_model.annotation.to_annotation()
                    ft = ft_model.to_feature(self.feature, roi, annotation)
                    features.append(ft)
                roi.features = features
            else:
                roi = None

        if roi is None:
            msg = f"No ROI stored with id={roi_id}."
            raise ValueError(msg)
        return roi

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
        ft_id = self._feature_id
        for roi in roi_list:
            if roi.features is not None:
                for ft in roi.features:
                    ft_model = FeatureModel(roi_id=roi.id, data=ft.to_str(), id=ft_id)
                    ann = ft.annotation
                    annotation_model = AnnotationModel(
                        id=ft_id,
                        sample_id=sample.id,
                        roi_id=roi.id,
                        label=ann.label,
                        charge=ann.charge,
                        isotopologue_label=ann.isotopologue_label,
                        isotopologue_index=ann.isotopologue_index,
                    )
                    ft_descriptors = {x: ft.get(x) for x in descriptor_names}

                    descriptor_model = self.DescriptorModel(
                        id=ft_id,
                        sample_id=sample.id,
                        roi_id=roi.id,
                        label=ann.label,
                        **ft_descriptors,
                    )

                    feature_model_list.append(ft_model)
                    annotation_model_list.append(annotation_model)
                    descriptor_model_list.append(descriptor_model)
                    ft_id += 1
        with self.SessionFactory() as session:
            session.add_all(feature_model_list)
            session.add_all(annotation_model_list)
            session.add_all(descriptor_model_list)
            session.commit()
            # only update the id if the transaction is successful
            self._feature_id = ft_id + 1

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

    def get_features_by_id(self, ft_id: int) -> Feature:
        """
        Retrieves all samples with an specific feature label.

        Parameters
        ----------
        ft_id : int

        Returns
        -------
        Feature

        """
        with self.SessionFactory() as session:
            stmt = select(FeatureModel).join(AnnotationModel).where(FeatureModel.id == ft_id)
            result = session.execute(stmt).scalar()
            if result is not None:
                annotation = result.annotation.to_annotation()
                roi = result.roi.to_roi(self.roi)
                feature = result.to_feature(self.feature, roi, annotation)

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

        descriptors.extend(["id", "roi_id", "sample_id", "label"])
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

    def get_sample_data(self, sample_id: str) -> SampleData:
        sample = self.search_sample(sample_id)
        roi_list = self.get_roi_list(sample)
        return SampleData(sample, roi_list)

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
        else:
            return result.to_sample()

    def store_sample_data(self, data: SampleData):
        self.add_roi_list(data.roi, data.sample)
        self.add_features(data.roi, data.sample)
