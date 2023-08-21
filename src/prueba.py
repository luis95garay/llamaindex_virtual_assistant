from dotenv import load_dotenv
from data_processing.data_processing_pipeline import PipelineDataProcessing

load_dotenv("credentials.env")

pipe = PipelineDataProcessing(source_type="web")
pipe(
    # path="https://mibluemedical.com/blue-medical-centro-de-especialidades/",
    path="https://mibluemedical.com/laboratorios/",
    name="Laboratorios",
    proceadure_type="add"
    )
