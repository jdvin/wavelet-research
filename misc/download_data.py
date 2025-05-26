import asyncio
import aiohttp
from pathlib import Path
import sys

file_ids = [
    "9hykz",
    "6rkbf",
    "ng49t",
    "9rb78",
    "2nzby",
    "jgrn8",
    "hpv6m",
    "7cmd4",
    "y9xfm",
    "fpvq8",
    "7hs4g",
    "4vshm",
    "pcm5k",
    "rj2y8",
    "3ebj9",
    "p7c9t",
]

# URL template for downloads
file_url = "https://osf.io/{file_id}/download"


async def download_file(session, file_id):
    """Download a single file asynchronously."""
    url = file_url.format(file_id=file_id)
    output_file = Path(f"/Volumes/T7/datasets/EEGEyeNet/downloaded_{file_id}")

    try:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"Failed to download {file_id}. Status: {response.status}")
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
                        print(f"{file_id}: {progress:.1f}%", end="\r", file=sys.stderr)

            print(f"\n{file_id}: Download complete")
            return True

    except Exception as e:
        print(f"\nError downloading {file_id}: {str(e)}")
        return False


async def download_all_files():
    """Download all files concurrently."""
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None)
    ) as session:
        tasks = [download_file(session, file_id) for file_id in file_ids]
        results = await asyncio.gather(*tasks)

        successful = sum(1 for r in results if r)
        print(f"\nDownloaded {successful} out of {len(file_ids)} files successfully")


if __name__ == "__main__":
    asyncio.run(download_all_files())
