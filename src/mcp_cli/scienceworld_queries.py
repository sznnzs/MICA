"""RNG-preserving public metadata queries for ScienceWorld."""
import hashlib
from .benchmarks.scienceworld import ScienceWorldAdapter
PROTOCOL = 'scienceworld-public-query-seeded.v2'
GETTERS = frozenset({'getPossibleActions', 'getPossibleActionsWithIDs', 'getPossibleObjects',
    'getPossibleObjectReferentLUTJSON', 'getPossibleObjectReferentTypesLUTJSON',
    'getValidActionObjectCombinations', 'getValidActionObjectCombinationsJSON',
    'getAllObjectIdsTypesReferentsLUTJSON'})

def java_random(adapter):
    jvm=adapter.environment._gateway.jvm
    singleton=getattr(getattr(jvm.scala.util,'Random$'),'MODULE$')
    return singleton.self()

def java_rng_bytes(adapter):
    jvm=adapter.environment._gateway.jvm
    buf=jvm.java.io.ByteArrayOutputStream();stream=jvm.java.io.ObjectOutputStream(buf)
    stream.writeObject(java_random(adapter));stream.flush()
    return str(jvm.java.util.Base64.getEncoder().encodeToString(buf.toByteArray()))

def restore_in_place(adapter,encoded):
    jvm=adapter.environment._gateway.jvm
    raw=jvm.java.util.Base64.getDecoder().decode(encoded)
    saved=jvm.java.io.ObjectInputStream(jvm.java.io.ByteArrayInputStream(raw)).readObject()
    target=java_random(adapter)
    for name in ['seed','haveNextNextGaussian','nextNextGaussian']:
        field=target.getClass().getDeclaredField(name);field.setAccessible(True)
        if name=='seed':
            # Py4J converts AtomicLong (a Number) to a Python integer. Undo
            # Random.setSeed's documented 48-bit scramble, retaining target.
            target.setSeed(int(field.get(saved)) ^ 0x5DEECE66D)
        elif name=='haveNextNextGaussian':field.setBoolean(target,field.getBoolean(saved))
        else:field.setDouble(target,field.getDouble(saved))
    assert java_rng_bytes(adapter)==encoded

def query_seed(environment_seed, method):
    if method not in GETTERS:
        raise ValueError('not a registered metadata getter')
    payload = f'{PROTOCOL}|{int(environment_seed)}|{method}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:6], 'big')

class PublicQueryProxy:
    def __init__(self, target, adapter):
        self.target, self.adapter = target, adapter

    def __getattr__(self, name):
        target = getattr(self.target, name)
        if name not in GETTERS:
            return target

        def query(*args, **kwargs):
            adapter = self.adapter
            before = java_rng_bytes(adapter)
            seed = query_seed(adapter.public_query_environment_seed, name)
            error = None
            try:
                java_random(adapter).setSeed(seed)
                return target(*args, **kwargs)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                restore_in_place(adapter, before)
                after = java_rng_bytes(adapter)
                adapter.public_query_audit.append({'method': name, 'query_seed': seed,
                    'rng_before_sha256': hashlib.sha256(before.encode()).hexdigest(),
                    'rng_after_sha256': hashlib.sha256(after.encode()).hexdigest(),
                    'rng_preserved': before == after, 'error_type': error,
                    'data_origin': 'current_execution'})
                if before != after:
                    raise RuntimeError('public metadata query changed the physics RNG')
        return query

class PublicQueryScienceWorldAdapter(ScienceWorldAdapter):
    def __init__(self, config):
        if config.get('public_query_protocol') != PROTOCOL:
            raise ValueError('explicit ScienceWorld public query protocol required')
        super().__init__(config)
        self.public_query_environment_seed = None
        self.public_query_audit = []

    def _environment(self):
        env = super()._environment()
        if not isinstance(env.server, PublicQueryProxy):
            env.server = PublicQueryProxy(env.server, self)
        return env

    def reset(self, task, *, environment_seed):
        self.public_query_environment_seed = int(environment_seed)
        self.scenario = str(task.metadata['task_name'])
        self.family = task.family
        self.split = task.split
        return super().reset(task, environment_seed=environment_seed)
