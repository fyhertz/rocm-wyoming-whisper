import argparse
import asyncio
import logging
from functools import partial

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from .const import WHISPER_LANGUAGES
import whisper
from .handler import FasterWhisperEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="medium",
        help="Name of whisper model to use (default medium)",
    )
    parser.add_argument(
        "--uri", help="unix:// or tcp://", default="tcp://0.0.0.0:10300"
    )
    parser.add_argument(
        "--download-dir",
        required=True,
        help="Directory to download models into",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use for inference (default: cuda)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Default language to set for transcription (default: en)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
    )

    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    _LOGGER.debug("Loading %s", args.model)
    whisper_model = whisper.load_model(
        args.model, device=args.device, download_root=args.download_dir
    )

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="whisper",
                description="Whisper",
                attribution=Attribution(
                    name="OpenAI",
                    url="https://github.com/openai/whisper",
                ),
                installed=True,
                models=[
                    AsrModel(
                        name=args.model,
                        description=args.model,
                        attribution=Attribution(
                            name="whisper",
                            url="https://github.com/openai/whisper",
                        ),
                        installed=True,
                        languages=WHISPER_LANGUAGES,
                    )
                ],
            )
        ],
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")
    model_lock = asyncio.Lock()
    await server.run(
        partial(
            FasterWhisperEventHandler,
            wyoming_info,
            args,
            whisper_model,
            model_lock,
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
