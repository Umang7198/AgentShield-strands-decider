class ApprovalError(Exception):
    """Something about this approval cannot be done."""


class ApprovalNotFound(ApprovalError):
    def __init__(self, approval_id: str) -> None:
        super().__init__(f"approval '{approval_id}' not found")
        self.approval_id = approval_id


class ApprovalNotPending(ApprovalError):
    def __init__(self, approval_id: str, status: str) -> None:
        super().__init__(f"approval '{approval_id}' is {status}, not pending")
        self.approval_id = approval_id
        self.status = status
