import requests
import os
from tqdm import tqdm

if __name__=="__main__":
    link = "https://storage.googleapis.com/kaggle-data-sets/1200301/2006107/compressed/PASCAL_VOC.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240304%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240304T051124Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=8e39eb85646bf0ae597063e07e02142a2421136fc9cf08466485eb661e96f1a7b7eb93c5e3f5a80417cde10973be79405575ef2a3e114d079316f63783cd1d88f8b9fdb68fca4d31e234307b1c7deafe9e6d648747a2996bf145c395d537ba032a4731c1db153c0b12aa794f28004b44b467f432751ad8f9e187ea30f26f48f544231f8359935081dd2b719a627ab94cdf5b11dc869c2d04424aecc6be6d795a884f6539264792841d9620ad6386067d285e57d719bdaf0273eb23fa1d6cccd7a9d8c0f557eb4c4c642a22f7814cb35233a93261708286651a060c9288e6a6cc1e2f7307fe6a54c0d56336285f5773f03cd416bcf04ecbe36a1784caa813deaf"
    response = requests.get(link, stream= True)
    
    total_size_in_bytes = int(response.headers.get('content-length',0))
    block_size = 1024
    progress_bar = tqdm(total= total_size_in_bytes, unit= "iB", unit_scale=True)

    file_path = os.path.join(".", "pascal_voc.zip")
    
    with open(file_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

    print(f"file downloaded of size {os.path.getsize(file_path)/10**9:.2f} gb")

    import zipfile
    extract_path = os.path.join('.','pascal_voc')

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        file_names = zip_ref.namelist()
        total_length = len(file_names)
        for file in tqdm(enumerate(file_names), total=total_length, desc="Extracting"):
            zip_ref.extract(member=file[1], path=extract_path)
