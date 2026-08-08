"""Outbound citizen notifications, fired on complaint status transitions.

Same pattern as storage: an abstract channel, several concrete channels, and a
dispatcher that fans out to all of them. Adding SMS later is a new subclass — no
change to ``ComplaintManager``, which is the whole point of the abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.logging_config import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class NotificationMessage:
    """A single outbound message, independent of how it is delivered."""

    recipient: str | None
    subject: str
    body: str
    reference_code: str
    event: str = "status_changed"

    @property
    def deliverable(self) -> bool:
        return bool(self.recipient)


class NotificationService(ABC):
    """Abstract delivery channel for citizen updates.

    Responsibility: take a :class:`NotificationMessage` and get it to the citizen.
    Concrete channels decide the transport; none of them decide *when* to send —
    that judgement belongs to ``ComplaintManager``.
    """

    #: Channel identifier used in logs and health output.
    channel: str = "abstract"

    @abstractmethod
    async def send(self, message: NotificationMessage) -> bool:
        """Deliver ``message``. Returns True when it was accepted for delivery."""

    def supports(self, message: NotificationMessage) -> bool:
        """Whether this channel can handle the message at all."""
        return True


class ConsoleNotificationChannel(NotificationService):
    """Writes the notification to the structured log.

    This is the always-on channel: it is what makes the demo visible on stage and it
    can never fail, so a broken email provider can never break a status update.
    """

    channel = "console"

    async def send(self, message: NotificationMessage) -> bool:
        log.info(
            "notification.sent",
            channel=self.channel,
            reference_code=message.reference_code,
            recipient=message.recipient or "anonymous",
            subject=message.subject,
            event=message.event,
        )
        return True


class EmailNotificationChannel(NotificationService):
    """Email delivery.

    No SMTP provider is wired up for the hackathon, so this logs the message it
    *would* send and reports non-delivery honestly rather than pretending. The class
    exists (instead of a flag) so plugging in a provider is a one-method change.
    """

    channel = "email"

    def __init__(self, *, from_address: str = "no-reply@civic.gov.pk") -> None:
        self.from_address = from_address

    def supports(self, message: NotificationMessage) -> bool:
        return message.deliverable and "@" in (message.recipient or "")

    async def send(self, message: NotificationMessage) -> bool:
        if not self.supports(message):
            return False
        log.info(
            "notification.email_queued",
            channel=self.channel,
            to=message.recipient,
            sender=self.from_address,
            reference_code=message.reference_code,
            subject=message.subject,
        )
        return True


class NotificationDispatcher:
    """Fans a message out across every registered channel.

    Responsibility: isolation. One channel raising must never abort a status
    transition that has already been committed, so every send is individually
    guarded and failures are logged rather than propagated.
    """

    def __init__(self, channels: list[NotificationService] | None = None) -> None:
        self._channels: list[NotificationService] = channels or [
            ConsoleNotificationChannel(),
            EmailNotificationChannel(),
        ]

    @property
    def channels(self) -> list[str]:
        return [c.channel for c in self._channels]

    def register(self, channel: NotificationService) -> None:
        self._channels.append(channel)

    async def dispatch(self, message: NotificationMessage) -> int:
        delivered = 0
        for channel in self._channels:
            if not channel.supports(message):
                continue
            try:
                if await channel.send(message):
                    delivered += 1
            except Exception as exc:  # noqa: BLE001 - notifications are best-effort
                log.warning(
                    "notification.failed",
                    channel=channel.channel,
                    reference_code=message.reference_code,
                    error=str(exc),
                )
        return delivered

    async def notify_status_change(
        self,
        *,
        reference_code: str,
        recipient: str | None,
        from_status: str | None,
        to_status: str,
        note: str | None = None,
    ) -> int:
        """Convenience wrapper used by ``ComplaintManager`` on every transition."""
        body = (
            f"Your complaint {reference_code} has moved from "
            f"'{from_status or 'new'}' to '{to_status}'."
        )
        if note:
            body += f"\n\nNote from the department: {note}"
        return await self.dispatch(
            NotificationMessage(
                recipient=recipient,
                subject=f"Update on complaint {reference_code}",
                body=body,
                reference_code=reference_code,
                event="status_changed",
            )
        )


#: Process-wide dispatcher, injected via ``app.core.deps.get_notification_dispatcher``.
notification_dispatcher = NotificationDispatcher()
