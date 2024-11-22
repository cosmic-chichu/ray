import os

from typing import Dict, Optional
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (EmbeddingRequest,
                                              EmbeddingResponse,
                                              ErrorResponse)
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_engine import BaseModelPath
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.logger import RequestLogger

logger = logging.getLogger("ray.serve")

app = FastAPI()


@serve.deployment(name="EmbedderDeployment")
@serve.ingress(app)
class EmbedderDeployment:
    def __init__(
            self,
            engine_args: AsyncEngineArgs,
            request_logger: Optional[RequestLogger] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_embedding = None
        self.engine_args = engine_args
        self.request_logger = request_logger
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app.post("/v1/embeddings")
    async def embed(
            self, request: EmbeddingRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP `end`point.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.openai_serving_embedding:
            model_config = await self.engine.get_model_config()
            # Determine the name of the served model for the OpenAI client.
            if self.engine_args.served_model_name is not None:
                BASE_MODEL_PATHS = [BaseModelPath(name=self.engine_args.served_model_name,
                                                  model_path=self.engine_args.served_model_name)]
            else:
                BASE_MODEL_PATHS = [BaseModelPath(name=self.engine_args.model, model_path=self.engine_args.model)]
            self.openai_serving_embedding = OpenAIServingEmbedding(
                self.engine,
                model_config,
                BASE_MODEL_PATHS,
                request_logger=self.request_logger,
                chat_template=None,
                chat_template_content_format=None,
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_embedding.create_embedding(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, EmbeddingResponse)
            return JSONResponse(content=generator.model_dump())


def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    parser = FlexibleArgumentParser(description="vLLM CLI")
    parser = make_arg_parser(parser)
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True
    engine_args.device = "cuda"
    engine_args.task = "embedding"

    return EmbedderDeployment.bind(
        engine_args,
        cli_args.get("request_logger"),
    )


model = build_app(
    {"model": os.environ['MODEL_ID'],
     "tensor-parallel-size": os.environ['TENSOR_PARALLELISM'],
     "pipeline-parallel-size": os.environ['PIPELINE_PARALLELISM'],
     })
