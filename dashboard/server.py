"""
FastAPI Dashboard Server for OI Gemini (Phase 2 Advanced Visualization).
Provides real-time WebSockets and REST API for the frontend dashboard.
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

try:
    import uvicorn
except ImportError:
    uvicorn = None
    logging.warning("uvicorn not available - dashboard server will not start")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# If we need access to AppManager, we'll need a way to share state.
# Since AppManager runs in a different thread/process context usually,
# we might use a singleton or pass it during startup if running in same process.
# Here we assume it's running in the same process but different thread.

app = FastAPI()
LOGGER = logging.getLogger("DashboardServer")

# Global reference to AppManager (will be injected)
_APP_MANAGER = None

def set_app_manager(manager):
    global _APP_MANAGER
    _APP_MANAGER = manager

# Connection Manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass # Handle disconnects gracefully

manager = ConnectionManager()

# --- Routes ---

@app.get("/api/status")
async def get_status():
    if _APP_MANAGER:
        return {"status": "online", "exchanges": list(_APP_MANAGER.exchange_handlers.keys())}
    return {"status": "offline"}

@app.get("/api/state/{exchange}")
async def get_exchange_state(exchange: str):
    if not _APP_MANAGER:
        return {"error": "AppManager not initialized"}
    
    handler = _APP_MANAGER.exchange_handlers.get(exchange)
    if not handler:
        return {"error": "Exchange not found"}
        
    # Snapshot of critical state
    return {
        "spot_price": handler.latest_spot_price,
        "future_price": handler.latest_future_price,
        "expiry": handler.expiry_date.isoformat() if handler.expiry_date else None,
        "positions": _APP_MANAGER.open_positions if hasattr(_APP_MANAGER, 'open_positions') else {}
    }

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection open, client handles pings
            # We can also push data here if we want per-client loops
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- Background Task for Broadcasting Updates ---

async def broadcast_state():
    """
    Polls AppManager state and broadcasts to WebSocket clients every second.
    """
    while True:
        if _APP_MANAGER:
            try:
                # Aggregate state from all exchanges
                payload = {
                    "timestamp": datetime.now().isoformat(),
                    "exchanges": {}
                }
                
                for ex_name, handler in _APP_MANAGER.exchange_handlers.items():
                    payload["exchanges"][ex_name] = {
                        "spot": handler.latest_spot_price,
                        "future": handler.latest_future_price,
                        "atm": handler.atm_strike,
                        # Add latest signal/sentiment if available
                    }
                
                # Check for sentiment
                if hasattr(_APP_MANAGER, 'last_nifty_sentiment_data'):
                    payload["sentiment"] = _APP_MANAGER.last_nifty_sentiment_data
                
                await manager.broadcast(json.dumps(payload))
                
            except Exception as e:
                LOGGER.error(f"Error broadcasting state: {e}")
        
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_state())

def run_dashboard_server(host="0.0.0.0", port=8000, app_manager=None):
    """
    Helper to run Uvicorn programmatically.
    Call this from a separate thread.
    """
    if uvicorn is None:
        LOGGER.error("uvicorn is not installed. Please install it: pip install uvicorn")
        return
    
    if app_manager:
        set_app_manager(app_manager)
    
    # Run uvicorn
    uvicorn.run(app, host=host, port=port, log_level="error")

