# -*- coding: utf-8 -*-
import os
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

_scheduler = None


def _daily_investor_email_job():
    """APScheduler job: generate fresh insights and email them."""
    logger.info("Scheduler: starting daily investor insights email job")
    try:
        # Import inside the function to avoid circular imports at module load
        from app import generate_investor_insights, investor_insights_cache
        from email_service import send_investor_email

        # Force fresh generation by clearing the cache
        investor_insights_cache['content'] = None
        investor_insights_cache['date']    = None
        investor_insights_cache['twitter_data'] = None

        content, generated_at = generate_investor_insights()

        if not content:
            logger.error("Scheduler: investor insights generation returned no content")
            return

        success, message = send_investor_email(content, generated_at)
        if success:
            logger.info(f"Scheduler: {message}")
        else:
            logger.error(f"Scheduler: email failed – {message}")

    except Exception as e:
        logger.exception(f"Scheduler: unhandled exception in daily job – {e}")


def start_scheduler():
    """
    Start the background scheduler.

    Reads INVESTOR_EMAIL_TIME from env (HH:MM, default 08:00).
    Safe to call multiple times – won't start a second instance.
    """
    global _scheduler

    if _scheduler and _scheduler.running:
        logger.debug("Scheduler already running, skipping start")
        return _scheduler

    email_time = os.environ.get('INVESTOR_EMAIL_TIME', '08:00')
    try:
        hour_str, minute_str = email_time.split(':')
        hour, minute = int(hour_str), int(minute_str)
    except (ValueError, AttributeError):
        logger.warning(f"Invalid INVESTOR_EMAIL_TIME '{email_time}', defaulting to 08:00")
        hour, minute = 8, 0

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        _daily_investor_email_job,
        trigger=CronTrigger(hour=hour, minute=minute),
        id='daily_investor_email',
        name='Daily Investor Insights Email',
        replace_existing=True,
        misfire_grace_time=3600,   # allow up to 1 h late if the process was down
    )
    _scheduler.start()
    logger.info(f"Scheduler started – daily investor email at {hour:02d}:{minute:02d}")
    return _scheduler


def stop_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
