class RequestModel:
    def __init__(self, request_id: str, user_id: str, request_data: dict):
        self.request_id = request_id
        self.user_id = user_id
        self.request_data = request_data

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "request_data": self.request_data
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            request_id=data.get("request_id"),
            user_id=data.get("user_id"),
            request_data=data.get("request_data", {})
        )