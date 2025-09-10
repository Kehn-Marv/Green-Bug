import os
import time
import pytest

from src.api.job_store import get_default_store


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get('REDIS_URL'), reason='REDIS_URL not configured')
async def test_redis_job_store_basic_flow():
    store = get_default_store()
    assert store is not None

    job_id = 'testjob' + str(int(time.time() * 1000))[-6:]

    # Create job
    await store.create_job(job_id, {'created_at': 'now'})

    # Publish some events
    await store.publish_event(job_id, {'status': 'running', 'progress': 10, 'message': 'start'})

    # Subscribe and read one event
    async for ev in store.subscribe(job_id):
        assert ev.get('progress') == 10
        break

    # Set result and fetch job
    await store.set_result(job_id, {'ok': True})
    job = await store.get_job(job_id)
    assert job.get('status') == 'completed' or job.get('result')
