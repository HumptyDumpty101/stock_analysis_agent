from fastapi import HTTPException

ERROR_CODES: dict[str, dict] = {
    # region Auth
    "AUTH001": {"status": 401, "message": "Sign in to use this feature."},
    "AUTH002": {"status": 403, "message": "You have used all 3 lifetime analysis credits. This is a portfolio demo — credits are intentionally limited."},
    # endregion Auth

    # region Input
    "INPUT001": {"status": 400, "message": "Message is required."},
    "INPUT002": {"status": 400, "message": "Message too long (max 300 characters)."},
    "INPUT003": {"status": 400, "message": "Invalid session ID."},
    # endregion Input

    # region Agent
    "AGNT001": {"status": 503, "message": "Agent unavailable."},
    "AGNT002": {"status": 504, "message": "Agent is unreachable."},
    "AGNT003": {"status": 500, "message": "Analysis failed. Please try again."},
    "AGNT004": {"status": 500, "message": "Stream interrupted."},
    # endregion Agent

    # region General
    "GEN001": {"status": 500, "message": "Something went wrong. Please try again."},
    # endregion General
}


class AppError(Exception):
    def __init__(self, code: str):
        entry = ERROR_CODES.get(code) or ERROR_CODES["GEN001"]
        self.code = code if code in ERROR_CODES else "GEN001"
        self.status_code = entry["status"]
        self.message = entry["message"]
        super().__init__(self.message)

    @property
    def json(self) -> dict:
        return {"code": self.code, "message": self.message}

    def raise_http(self) -> None:
        raise HTTPException(status_code=self.status_code, detail=self.json)
