"""FastAPI application with WebSocket support."""

import logging

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from mybot.core.context import SharedContext

logger = logging.getLogger(__name__)

# Default CORS origins when none configured — restrict to localhost only
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]


def create_app(context: SharedContext) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MyBot WebSocket Server",
        description="WebSocket server for real-time agent communication",
        version="0.1.0",
    )
    app.state.context = context

    # Enable CORS for web clients — use configured origins or safe defaults
    cors_origins = getattr(context.config.api, "cors_origins", None)
    if not cors_origins:
        cors_origins = _DEFAULT_CORS_ORIGINS
        logger.info(
            "No 'api.cors_origins' configured — defaulting to localhost only. "
            "Set 'api.cors_origins' in config.user.yaml to allow other origins."
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time event streaming and chat."""
        # Optional token-based authentication
        ws_token = getattr(context.config.api, "ws_auth_token", None)
        if ws_token:
            client_token = websocket.query_params.get("token")
            if client_token != ws_token:
                await websocket.close(code=4003, reason="Authentication required")
                return

        await websocket.accept()

        # Check if WebSocket worker is available
        if context.websocket_worker is None:
            await websocket.close(code=1013, reason="WebSocket not available")
            return

        # Hand off to worker
        await context.websocket_worker.handle_connection(websocket)

    return app
