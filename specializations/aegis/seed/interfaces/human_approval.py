"""
Human Approval Interface
All critical decisions require explicit human approval
"""

import json
import hashlib
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of an approval request"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ChangeType(Enum):
    """Types of changes requiring approval"""
    ARCHITECTURE_MODIFICATION = "architecture_modification"
    CODE_GENERATION = "code_generation"
    TRAINING_PARAMETER_CHANGE = "training_parameter_change"
    CAPABILITY_DEPLOYMENT = "capability_deployment"
    SAFETY_BOUND_MODIFICATION = "safety_bound_modification"
    EMERGENCY_OVERRIDE = "emergency_override"


@dataclass
class ApprovalRequest:
    """Request for human approval"""
    request_id: str
    change_type: ChangeType
    title: str
    description: str
    rationale: str
    risk_assessment: Dict[str, Any]
    proposed_changes: Dict[str, Any]
    reversibility: bool
    estimated_impact: str
    created_at: datetime
    expires_at: datetime
    status: ApprovalStatus
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_notes: Optional[str] = None
    approval_code: Optional[str] = None


class ApprovalManager:
    """Manages human approval workflow"""

    def __init__(self, approval_timeout_hours: int = 24):
        self.approval_timeout_hours = approval_timeout_hours
        self.pending_requests: Dict[str, ApprovalRequest] = {}
        self.request_history: List[ApprovalRequest] = []

        # Callbacks for notifications
        self.notification_callbacks = []
        self.on_approval_callbacks = []  # Called when request approved

    def register_notification_callback(self, callback):
        """Register a callback for approval notifications"""
        self.notification_callbacks.append(callback)

    def register_approval_callback(self, callback):
        """
        Register a callback to be called when a request is approved

        Callback signature: callback(approved_request: ApprovalRequest) -> None
        """
        self.on_approval_callbacks.append(callback)

    def request_approval(
        self,
        change_type: ChangeType,
        title: str,
        description: str,
        rationale: str,
        risk_assessment: Dict[str, Any],
        proposed_changes: Dict[str, Any],
        reversibility: bool = True,
        estimated_impact: str = "medium"
    ) -> str:
        """
        Create an approval request

        Returns:
            request_id for tracking
        """

        request_id = str(uuid.uuid4())

        request = ApprovalRequest(
            request_id=request_id,
            change_type=change_type,
            title=title,
            description=description,
            rationale=rationale,
            risk_assessment=risk_assessment,
            proposed_changes=proposed_changes,
            reversibility=reversibility,
            estimated_impact=estimated_impact,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=self.approval_timeout_hours),
            status=ApprovalStatus.PENDING
        )

        self.pending_requests[request_id] = request

        # Send notification
        self._notify_human_operator(request)

        logger.info(
            f"Approval request created: {request_id} - {title}"
        )

        return request_id

    def check_approval_status(self, request_id: str) -> Optional[ApprovalStatus]:
        """Check if a request has been approved"""

        if request_id not in self.pending_requests:
            # Check history
            for req in self.request_history:
                if req.request_id == request_id:
                    return req.status
            return None

        request = self.pending_requests[request_id]

        # Check if expired
        if datetime.now() > request.expires_at and request.status == ApprovalStatus.PENDING:
            request.status = ApprovalStatus.EXPIRED
            self._move_to_history(request_id)
            logger.warning(f"Approval request {request_id} expired")

        return request.status

    def approve_request(
        self,
        request_id: str,
        reviewer_name: str,
        approval_code: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Approve a pending request

        Args:
            request_id: ID of request to approve
            reviewer_name: Name of human reviewer
            approval_code: Unique approval code for audit trail
            notes: Optional reviewer notes

        Returns:
            True if approved successfully
        """

        if request_id not in self.pending_requests:
            logger.error(f"Approval request {request_id} not found")
            return False

        request = self.pending_requests[request_id]

        if request.status != ApprovalStatus.PENDING:
            logger.error(
                f"Cannot approve request {request_id} with status {request.status}"
            )
            return False

        # Update request
        request.status = ApprovalStatus.APPROVED
        request.reviewed_by = reviewer_name
        request.reviewed_at = datetime.now()
        request.review_notes = notes
        request.approval_code = approval_code

        # Move to history
        self._move_to_history(request_id)

        logger.info(
            f"Request {request_id} approved by {reviewer_name} "
            f"with code {approval_code[:8]}..."
        )

        # Trigger approval callbacks
        logger.info(f"Triggering {len(self.on_approval_callbacks)} approval callbacks...")
        for i, callback in enumerate(self.on_approval_callbacks):
            try:
                logger.info(f"  Executing callback {i+1}: {callback.__name__}")
                callback(request)
                logger.info(f"  ✓ Callback {i+1} completed successfully")
            except Exception as e:
                logger.error(f"✗ Error in approval callback {i+1}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        return True

    def reject_request(
        self,
        request_id: str,
        reviewer_name: str,
        reason: str
    ) -> bool:
        """
        Reject a pending request

        Returns:
            True if rejected successfully
        """

        if request_id not in self.pending_requests:
            logger.error(f"Approval request {request_id} not found")
            return False

        request = self.pending_requests[request_id]

        if request.status != ApprovalStatus.PENDING:
            logger.error(
                f"Cannot reject request {request_id} with status {request.status}"
            )
            return False

        # Update request
        request.status = ApprovalStatus.REJECTED
        request.reviewed_by = reviewer_name
        request.reviewed_at = datetime.now()
        request.review_notes = reason

        # Move to history
        self._move_to_history(request_id)

        logger.info(
            f"Request {request_id} rejected by {reviewer_name}: {reason}"
        )

        return True

    def wait_for_approval(
        self,
        request_id: str,
        check_interval_seconds: int = 60
    ) -> bool:
        """
        Block until approval is granted or denied

        Returns:
            True if approved, False if rejected/expired
        """

        import time

        while True:
            status = self.check_approval_status(request_id)

            if status == ApprovalStatus.APPROVED:
                return True
            elif status in [ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED]:
                return False
            elif status is None:
                logger.error(f"Request {request_id} not found")
                return False

            time.sleep(check_interval_seconds)

    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending approval requests"""
        return list(self.pending_requests.values())

    def get_request_details(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a request"""

        # Check pending
        if request_id in self.pending_requests:
            return asdict(self.pending_requests[request_id])

        # Check history
        for req in self.request_history:
            if req.request_id == request_id:
                return asdict(req)

        return None

    def _move_to_history(self, request_id: str):
        """Move request from pending to history"""

        if request_id in self.pending_requests:
            request = self.pending_requests.pop(request_id)
            self.request_history.append(request)

    def _notify_human_operator(self, request: ApprovalRequest):
        """Send notification to human operator"""

        message = self._format_approval_request(request)

        for callback in self.notification_callbacks:
            try:
                callback(message, request)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")

    def _format_approval_request(self, request: ApprovalRequest) -> str:
        """Format approval request as human-readable message"""

        return f"""
╔══════════════════════════════════════════════════════════════╗
║              HUMAN APPROVAL REQUIRED                          ║
╚══════════════════════════════════════════════════════════════╝

Request ID: {request.request_id}
Type: {request.change_type.value}

Title: {request.title}

Description:
{request.description}

Rationale:
{request.rationale}

Risk Assessment:
{json.dumps(request.risk_assessment, indent=2)}

Proposed Changes:
{json.dumps(request.proposed_changes, indent=2)}

Reversibility: {'Yes' if request.reversibility else 'No'}
Estimated Impact: {request.estimated_impact}

Created: {request.created_at.isoformat()}
Expires: {request.expires_at.isoformat()}

═══════════════════════════════════════════════════════════════

To approve this request, use:
  approval_manager.approve_request(
      request_id="{request.request_id}",
      reviewer_name="<your_name>",
      approval_code="<unique_code>",
      notes="<optional_notes>"
  )

To reject this request, use:
  approval_manager.reject_request(
      request_id="{request.request_id}",
      reviewer_name="<your_name>",
      reason="<rejection_reason>"
  )

═══════════════════════════════════════════════════════════════
"""

    def generate_approval_report(self) -> Dict[str, Any]:
        """Generate summary report of all approvals"""

        total_requests = len(self.request_history) + len(self.pending_requests)
        approved = sum(1 for r in self.request_history if r.status == ApprovalStatus.APPROVED)
        rejected = sum(1 for r in self.request_history if r.status == ApprovalStatus.REJECTED)
        expired = sum(1 for r in self.request_history if r.status == ApprovalStatus.EXPIRED)
        pending = len(self.pending_requests)

        return {
            'total_requests': total_requests,
            'approved': approved,
            'rejected': rejected,
            'expired': expired,
            'pending': pending,
            'approval_rate': approved / total_requests if total_requests > 0 else 0,
            'pending_requests': [
                {
                    'id': r.request_id,
                    'title': r.title,
                    'type': r.change_type.value,
                    'created': r.created_at.isoformat(),
                    'expires': r.expires_at.isoformat()
                }
                for r in self.pending_requests.values()
            ]
        }


# Example notification function (can be customized)
def console_notification(message: str, request: ApprovalRequest):
    """Print approval request to console"""
    print("\n" + "="*70)
    print(message)
    print("="*70 + "\n")


def email_notification(message: str, request: ApprovalRequest):
    """
    Send email notification (placeholder)
    In production, integrate with actual email service
    """
    # This would integrate with an email service
    logger.info(f"Email notification would be sent for request {request.request_id}")


def slack_notification(message: str, request: ApprovalRequest):
    """
    Send Slack notification (placeholder)
    In production, integrate with Slack API
    """
    # This would integrate with Slack
    logger.info(f"Slack notification would be sent for request {request.request_id}")
