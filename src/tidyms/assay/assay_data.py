from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Type, Union
from pathlib import Path
from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import declarative_base, sessionmaker, Mapped, mapped_column
from sqlalchemy.exc import IntegrityError
from ..lcms import Feature, Roi

# TODO: get sample by processing stage
# TODO: get ROI with and without features.
# TODO: Idea: Assays are created using create_assay and load_assay functions.


@dataclass
class Sample:
    path: Union[Path, str]
    id: str
    ms_level: int = 1
    start: Optional[float] = None
    end: Optional[float] = None
    group: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)

        if self.id is None:
            self.id = self.path.stem

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(d["path"])
        return d


class AssayData:

    Base = declarative_base()

    class SampleModel(Base):
        __tablename__ = "samples"
        id: Mapped[str] = mapped_column(String, primary_key=True)
        path: Mapped[str] = mapped_column(String, nullable=False)
        ms_level: Mapped[int] = mapped_column(Integer)
        start: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        end: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        group: Mapped[str] = mapped_column(String, nullable=True)

    class RoiModel(Base):
        __tablename__ = "rois"
        sample_id: Mapped[str] = mapped_column(
            String, ForeignKey("samples.id"), primary_key=True
        )
        roi_id: Mapped[int] = mapped_column(Integer, primary_key=True)
        data: Mapped[str] = mapped_column(String)

    def __init__(self, name: Union[Path, str], roi: Type[Roi], feature: Type[Feature]):

        self.roi = roi
        self.feature = feature

        # Create a database engine and session factory
        db_address = f"sqlite:///{name}.db"
        self.engine = create_engine(db_address)
        self.SessionFactory = sessionmaker(bind=self.engine)

        # Create the table in the database
        self.Base.metadata.create_all(self.engine)

    def add_samples(self, samples: list[Sample]):
        samples = [self.SampleModel(**x.to_dict()) for x in samples]
        with self.SessionFactory() as session:
            try:
                session.add_all(samples)
                session.commit()
            except IntegrityError:
                msg = "Trying to insert a sample with an existing ID"
                raise ValueError(msg)

    def get_samples(self) -> list[Sample]:
        with self.SessionFactory() as session:
            stmt = select(self.SampleModel)
            result = session.execute(stmt)
            samples = list()
            for row in result:
                sm = row.SampleModel
                sample = Sample(Path(sm.path), sm.id, sm.ms_level, sm.start, sm.end, sm.group)
                samples.append(sample)
            return samples

    def delete_samples(self, sample_ids: list[str]) -> None:
        with self.SessionFactory() as session:
            stmt = delete(self.SampleModel).where(self.SampleModel.id.in_(sample_ids))
            session.execute(stmt)
            session.commit()

    def add_roi_list(self, roi_list: Sequence[Roi], sample: Sample):

        roi_model_list = list()
        for roi in roi_list:
            roi_model = AssayData.RoiModel(
                sample_id=sample.id, roi_id=roi.index, data=roi.to_string()
            )
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
            stmt = select(self.RoiModel).where(self.RoiModel.sample_id == sample.id)
            result = session.execute(stmt)
            roi_list = list()
            for row in result:
                rm = row.RoiModel
                roi = self.roi.from_string(rm.data)
                roi_list.append(roi)
            return roi_list
    
    def delete_roi_list(self, sample: Sample):
        with self.SessionFactory() as session:
            stmt = delete(self.RoiModel).where(self.RoiModel.sample_id == sample.id)
            session.execute(stmt)
            session.commit()
