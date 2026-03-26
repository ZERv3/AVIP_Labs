import requests

ORIGIN = "https://www.slavcorpora.ru"
SAMPLE_ID = "b008ae91-32cf-4d7d-84e4-996144e4edb7"


def get_sample_image_paths(origin: str = ORIGIN, sample_id: str = SAMPLE_ID) -> list[str]:
    sample = requests.get(f"{origin}/api/samples/{sample_id}").json()
    return [f"{origin}/images/{p['filename']}" for p in sample["pages"]]
