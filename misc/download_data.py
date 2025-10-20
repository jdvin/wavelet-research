import asyncio
import os
import sys
from pathlib import Path
import argparse

import aiohttp
from google.cloud import storage

url = "https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/"
# path = "EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/{sub}/RSEEG/{file}"
path = "EEG_MPILMBB_LEMON/EEG_Localizer_BIDS_ID/{sub}/{file}"
# local_root = "/Volumes/T7/datasets/"
local_root = "/teamspace/studios/this_studio/wavelet-research/data/"

# files = [".eeg", ".vhdr", ".vmrk"]
files = [".mat"]

subjects = [
    "sub-010002",
    "sub-010003",
    "sub-010004",
    "sub-010005",
    "sub-010006",
    "sub-010007",
    "sub-010010",
    "sub-010012",
    "sub-010015",
    "sub-010016",
    "sub-010017",
    "sub-010019",
    "sub-010020",
    "sub-010021",
    "sub-010022",
    "sub-010023",
    "sub-010024",
    "sub-010026",
    "sub-010027",
    "sub-010028",
    "sub-010029",
    "sub-010030",
    "sub-010031",
    "sub-010032",
    "sub-010033",
    "sub-010034",
    "sub-010035",
    "sub-010036",
    "sub-010037",
    "sub-010038",
    "sub-010039",
    "sub-010040",
    "sub-010041",
    "sub-010042",
    "sub-010044",
    "sub-010045",
    "sub-010046",
    "sub-010047",
    "sub-010048",
    "sub-010049",
    "sub-010050",
    "sub-010051",
    "sub-010052",
    "sub-010053",
    "sub-010056",
    "sub-010059",
    "sub-010060",
    "sub-010061",
    "sub-010062",
    "sub-010063",
    "sub-010064",
    "sub-010065",
    "sub-010066",
    "sub-010067",
    "sub-010068",
    "sub-010069",
    "sub-010070",
    "sub-010071",
    "sub-010072",
    "sub-010073",
    "sub-010074",
    "sub-010075",
    "sub-010076",
    "sub-010077",
    "sub-010078",
    "sub-010079",
    "sub-010080",
    "sub-010081",
    "sub-010083",
    "sub-010084",
    "sub-010085",
    "sub-010086",
    "sub-010087",
    "sub-010088",
    "sub-010089",
    "sub-010090",
    "sub-010091",
    "sub-010092",
    "sub-010093",
    "sub-010094",
    "sub-010100",
    "sub-010104",
    "sub-010126",
    "sub-010134",
    "sub-010136",
    "sub-010137",
    "sub-010138",
    "sub-010141",
    "sub-010142",
    "sub-010146",
    "sub-010148",
    "sub-010150",
    "sub-010152",
    "sub-010155",
    "sub-010157",
    "sub-010162",
    "sub-010163",
    "sub-010164",
    "sub-010165",
    "sub-010166",
    "sub-010168",
    "sub-010170",
    "sub-010176",
    "sub-010183",
    "sub-010191",
    "sub-010192",
    "sub-010193",
    "sub-010194",
    "sub-010195",
    "sub-010196",
    "sub-010197",
    "sub-010199",
    "sub-010200",
    "sub-010201",
    "sub-010202",
    "sub-010203",
    "sub-010204",
    "sub-010207",
    "sub-010210",
    "sub-010213",
    "sub-010214",
    "sub-010215",
    "sub-010216",
    "sub-010218",
    "sub-010219",
    "sub-010220",
    "sub-010222",
    "sub-010223",
    "sub-010224",
    "sub-010226",
    "sub-010227",
    "sub-010228",
    "sub-010230",
    "sub-010231",
    "sub-010232",
    "sub-010233",
    "sub-010234",
    "sub-010235",
    "sub-010236",
    "sub-010237",
    "sub-010238",
    "sub-010239",
    "sub-010240",
    "sub-010241",
    "sub-010242",
    "sub-010243",
    "sub-010244",
    "sub-010245",
    "sub-010246",
    "sub-010247",
    "sub-010248",
    "sub-010249",
    "sub-010250",
    "sub-010251",
    "sub-010252",
    "sub-010254",
    "sub-010255",
    "sub-010256",
    "sub-010257",
    "sub-010258",
    "sub-010259",
    "sub-010260",
    "sub-010261",
    "sub-010262",
    "sub-010263",
    "sub-010264",
    "sub-010265",
    "sub-010266",
    "sub-010267",
    "sub-010268",
    "sub-010269",
    "sub-010270",
    "sub-010271",
    "sub-010272",
    "sub-010273",
    "sub-010274",
    "sub-010275",
    "sub-010276",
    "sub-010277",
    "sub-010278",
    "sub-010279",
    "sub-010280",
    "sub-010281",
    "sub-010282",
    "sub-010283",
    "sub-010284",
    "sub-010285",
    "sub-010286",
    "sub-010287",
    "sub-010288",
    "sub-010289",
    "sub-010290",
    "sub-010291",
    "sub-010292",
    "sub-010293",
    "sub-010294",
    "sub-010295",
    "sub-010296",
    "sub-010297",
    "sub-010298",
    "sub-010299",
    "sub-010300",
    "sub-010301",
    "sub-010302",
    "sub-010303",
    "sub-010304",
    "sub-010305",
    "sub-010306",
    "sub-010307",
    "sub-010308",
    "sub-010309",
    "sub-010310",
    "sub-010311",
    "sub-010314",
    "sub-010315",
    "sub-010316",
    "sub-010317",
    "sub-010318",
    "sub-010319",
    "sub-010321",
]

parser = argparse.ArgumentParser()
parser.add_argument("bucket")
gcs_bucket_name = parser.parse_args().bucket


async def upload_and_cleanup(bucket, local_path, blob_path):
    """Upload the downloaded file to GCS and remove the local copy."""
    loop = asyncio.get_running_loop()
    local_path = Path(local_path)

    def _upload_then_delete():
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(local_path))
        local_path.unlink()

    await loop.run_in_executor(None, _upload_then_delete)


async def download_file(
    session, base_url, path_template, local_root, file_suffix, subject, bucket, prefix
):
    """Download a single file asynchronously and mirror it to GCS."""
    file_id = subject + file_suffix
    target_path = path_template.format(sub=subject, file=file_id)
    file_url = base_url + target_path
    output_file = Path(local_root) / target_path
    os.makedirs(output_file.parent, exist_ok=True)

    try:
        async with session.get(file_url) as response:
            if response.status != 200:
                print(f"Failed to download {file_url}. Status: {response.status}")
                return False

            total_size = int(response.headers.get("content-length", 0))
            downloaded_size = 0

            with open(output_file, "wb") as f:
                async for chunk in response.content.iter_chunked(8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        progress = (
                            (downloaded_size / total_size) * 100 if total_size else 0
                        )
                        print(
                            f"{file_id}: {progress:.1f}%",
                            end="\r",
                            file=sys.stderr,
                        )

        print(f"\n{file_id}: Download complete")
    except Exception as e:
        print(f"\nError downloading {file_id}: {str(e)}")
        return False

    blob_path = "/".join(part for part in (prefix, target_path) if part)
    try:
        await upload_and_cleanup(bucket, output_file, blob_path)
        print(
            f"{file_id}: Uploaded to gs://{bucket.name}/{blob_path} and removed local copy"
        )
        return True
    except Exception as e:
        print(f"Error uploading {file_id} to GCS: {str(e)}")
        return False


async def download_all_files():
    """Download all files concurrently and mirror them to GCS."""
    gcs_prefix = os.environ.get("GCS_PREFIX", "").strip("/")
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket_name)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None)
    ) as session:
        tasks = []
        for subject in subjects:
            for file_suffix in files:
                tasks.append(
                    download_file(
                        session,
                        url,
                        path,
                        local_root,
                        file_suffix,
                        subject,
                        bucket,
                        gcs_prefix,
                    )
                )
        results = await asyncio.gather(*tasks)

        successful = sum(1 for r in results if r)
        print(f"\nMirrored {successful} out of {len(tasks)} files successfully")


if __name__ == "__main__":
    asyncio.run(download_all_files())
