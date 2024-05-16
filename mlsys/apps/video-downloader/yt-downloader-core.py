#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path

import yt_dlp


logger = logging.getLogger(__name__)


def download_video(
    url: str,
    download_dir: Path = Path("."),
    max_resolution: int = 720,
    cookie_file: str = "",
    verbose: bool = True,
):
    ydl_opts = {
        "format": f"best[height<={max_resolution}][ext=mp4]",
        "quiet": (not verbose),
        "no_warnings": (not verbose),
    }
    if cookie_file:
        ydl_opts["cookiefile"] = cookie_file
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        video_info = ydl.extract_info(url, download=False)
        if not isinstance(video_info, dict):
            logger.error(f"video_info is not a dict??? type={type(video_info)}")
            return
        if "title" not in video_info or "ext" not in video_info:
            logger.error("id or ext not in the video_info dict???")
            return
    # process the title
    title_items = [x.lower() for x in video_info["title"].split()[:10]]
    filename = "_".join(title_items)
    # save the metadata info
    metadata_file = download_dir / f"{filename}.json"
    with open(metadata_file, "wt") as fp:
        json.dump(video_info, fp, indent=4)
    # downloada the video
    video_file = download_dir / f"{filename}.{video_info['ext']}"
    ydl_opts["outtmpl"] = video_file.as_posix()
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        if not video_file or not video_file.exists():
            logger.error(f"Could not find local file {video_file.as_posix()} after download")
            return
    logger.info(f"Downloaded video at {video_file}")


def main(args):
    download_video(args.url, Path(args.output))


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Download youtube videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        help="URL of the youtube video to download",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory",
        required=True,
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
