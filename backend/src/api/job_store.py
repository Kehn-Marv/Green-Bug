import asyncio
import os
import json
from typing import Any, AsyncGenerator, Dict, Optional
from types import ModuleType

try:
    import redis.asyncio as aioredis
except Exception:
    aioredis = None

class InMemoryJobStore:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.queues: Dict[str, asyncio.Queue] = {}

    async def create_job(self, job_id: str, metadata: Dict[str, Any]):
        self.jobs[job_id] = dict(metadata)
        self.jobs[job_id].setdefault('status', 'queued')
        self.jobs[job_id].setdefault('progress', 0)
        self.queues[job_id] = asyncio.Queue()

    async def publish_event(self, job_id: str, event: Dict[str, Any]):
        # update metadata fields if present
        if job_id in self.jobs:
            if 'status' in event:
                self.jobs[job_id]['status'] = event['status']
            if 'progress' in event:
                self.jobs[job_id]['progress'] = event['progress']
            if 'result' in event:
                self.jobs[job_id]['result'] = event['result']
            if 'error' in event:
                self.jobs[job_id]['error'] = event['error']
        q = self.queues.get(job_id)
        if q:
            await q.put(event)

    async def set_result(self, job_id: str, result: Dict[str, Any]):
        await self.publish_event(job_id, {'status': 'completed', 'progress': 100, 'result': result})

    async def set_failed(self, job_id: str, error: str):
        await self.publish_event(job_id, {'status': 'failed', 'error': error})

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    async def subscribe(self, job_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        q = self.queues.setdefault(job_id, asyncio.Queue())
        # yield current state first
        job = self.jobs.get(job_id, {})
        yield {'status': job.get('status', 'queued'), 'progress': job.get('progress', 0)}
        while True:
            ev = await q.get()
            yield ev
            if ev.get('status') in ('completed', 'failed'):
                break


class RedisJobStore:
    def __init__(self, redis_url: str):
        if aioredis is None:
            raise RuntimeError('redis.asyncio not available')
        self.redis = aioredis.from_url(redis_url)

    async def create_job(self, job_id: str, metadata: Dict[str, Any]):
        key = f'job:{job_id}:meta'
        await self.redis.hset(key, mapping={k: json.dumps(v) for k, v in metadata.items()})
        await self.redis.hset(key, 'status', 'queued')
        await self.redis.hset(key, 'progress', '0')

    async def publish_event(self, job_id: str, event: Dict[str, Any]):
        channel = f'job:{job_id}:events'
        # store relevant fields in meta hash
        key = f'job:{job_id}:meta'
        if 'status' in event:
            await self.redis.hset(key, 'status', event['status'])
        if 'progress' in event:
            await self.redis.hset(key, 'progress', str(event['progress']))
        if 'result' in event:
            await self.redis.hset(key, 'result', json.dumps(event['result']))
        if 'error' in event:
            await self.redis.hset(key, 'error', event['error'])
        await self.redis.publish(channel, json.dumps(event))

    async def set_result(self, job_id: str, result: Dict[str, Any]):
        await self.publish_event(job_id, {'status': 'completed', 'progress': 100, 'result': result})

    async def set_failed(self, job_id: str, error: str):
        await self.publish_event(job_id, {'status': 'failed', 'error': error})

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        key = f'job:{job_id}:meta'
        raw = await self.redis.hgetall(key)
        if not raw:
            return None
        parsed = {}
        for k, v in raw.items():
            try:
                parsed[k.decode() if isinstance(k, bytes) else k] = json.loads(v)
            except Exception:
                parsed[k.decode() if isinstance(k, bytes) else k] = v.decode() if isinstance(v, bytes) else v
        return parsed

    async def subscribe(self, job_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        channel = f'job:{job_id}:events'
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(channel)
        # send current state
        meta = await self.get_job(job_id) or {}
        yield {'status': meta.get('status', 'queued'), 'progress': int(meta.get('progress', 0))}
        try:
            while True:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=None)
                if msg is None:
                    await asyncio.sleep(0.1)
                    continue
                data = msg['data']
                if isinstance(data, bytes):
                    try:
                        ev = json.loads(data.decode())
                    except Exception:
                        ev = {'data': data.decode(errors='ignore')}
                else:
                    ev = data
                yield ev
                if ev.get('status') in ('completed', 'failed'):
                    break
        finally:
            await pubsub.unsubscribe(channel)


def get_default_store():
    redis_url = os.environ.get('REDIS_URL')
    if redis_url and aioredis is not None:
        return RedisJobStore(redis_url)
    return InMemoryJobStore()


# Allow tests to override the default store
_override_store = None

def set_default_store_for_tests(store):
    """Set a specific job store instance (used by tests to inject InMemoryJobStore)."""
    global _override_store
    _override_store = store


def get_store():
    """Return either the overridden store (for tests) or the default store.

    This is intended to be a stable accessor used by application code.
    """
    if _override_store is not None:
        return _override_store
    return get_default_store()
