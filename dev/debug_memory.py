"""
Experiment Script Related to Pytorch Memory Leak Issue

References:
    https://github.com/pytorch/pytorch/issues/13246
    https://gist.github.com/mprostock/2850f3cd465155689052f0fa3a177a50

Potential Solutions:
    https://stackoverflow.com/questions/6832554/multiprocessing-how-do-i-share-a-dict-among-multiple-processes


Notes:
    The issue does not stem from any quirk in the multiprocessing library. It
    is a fundamental consequence of Python reference counting and the
    operating-system level fork operation. When the OS forks the base Python
    process it creates a new nearly identical process (Python variables even
    have the same id). This new process is very lightweight because it does not
    copy over all the memory from the original program. Instead, it will only
    copy bits of memory as they are changed, i.e. diverge from the base
    process. This is the copy-on-write behavior. When an item of a Python list
    is accessed by the forked process, it must increment the reference count of
    whatever it accessed, and thus the OS perceives a write and triggers the
    copy on write. But the OS doesn't just copy the small bit of memory that
    was touched. It copies the entire memory page that the reference count for
    the variable existed on. That's why the problem is so much worse when you
    do random access (in sequential access the memory page that is copied
    likely has the next reference count you were going to increment anyway, but
    in random access discontiguous blocks of memory are copied,... well...
    randomly). The one part I don't have a firm grasp on is why the problem
    doesn't plateau as you start to randomly access information in pages you
    already copied.  Perhaps the information is stale somehow? I'm not sure.
    But that is my best understanding of the issue.

    Using a pointer to a database like SQLite completely side-steps this
    problem, because the only information that is forked is a string that
    points to the database URI. New connections are opened up in each of the
    forked processes. The only issue I've had is accessing a row is now O(N
    log(N)) instead of O(1). This can be mitigated with memoized caching, which
    again for a reason I don't entirely understand, uses less memory than
    fork's copy-on-write behavior. However, I see speed benefits of SQL when I
    scale from 10,000 to 100,000 images. The SQL+memoized cache backend was
    running consistently at 45Hz as I scaled up (theoretically there should be
    a logarithmic slowdown, but it appears to be small enough effect that I
    didn't see it), whereas the in-memory json data structure starts at over
    100Hz, but slows down to 1.1Hz at scale (which theoretically should have
    been constant at scale, but that copy-on-write appears to add a lot of
    overhead).
"""
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import psutil
import ubelt as ub
import sys


class CustomDataset(Dataset):
    def __init__(self, storage_mode='numpy', return_mode='tensor', total=24e7):
        self.return_mode = return_mode
        self.storage_mode = storage_mode

        assert self.return_mode in {'tensor', 'dict', 'tuple', 'list'}

        if storage_mode == 'numpy':
            self.data = np.array([x for x in range(int(total))])
        elif storage_mode == 'python':
            self.data = [x for x in range(int(total))]
        elif storage_mode == 'ndsampler-sql':
            import ndsampler
            import kwcoco
            from kwcoco.coco_sql_dataset import ensure_sql_coco_view
            dset = kwcoco.CocoDataset.demo(
                'vidshapes', num_videos=1, num_frames=total,
                gsize=(64, 64)
            )
            dset = ensure_sql_coco_view(dset)
            print('dset.uri = {!r}'.format(dset.uri))
            dset.hashid = 'fake-hashid'
            sampler = ndsampler.CocoSampler(dset, backend=None)
            self.data = sampler
            # sampler.load_item(0)
            # tr = sampler.regions.get_item(0)
            # sampler.load_sample(tr)
            # assert total <= 1000
            # sampler = ndsampler.CocoSampler.demo('shapes{}'.format(total))
            # sampler = ndsampler.CocoSampler.demo('shapes{}'.format(total))
        elif storage_mode == 'ndsampler':
            import ndsampler
            # assert total <= 10000
            sampler = ndsampler.CocoSampler.demo(
                'vidshapes', num_videos=1, num_frames=total,
                gsize=(64, 64)
            )
            self.data = sampler
        else:
            raise KeyError(storage_mode)

    def __len__(self):
        return len(self.data)

    # def __getstate__(self):
    #     print('\n\nGETTING CUSTOM DATASET STATE')
    #     return super().__getstate__()

    # def __setstate__(self, val):
    #     print('\n\nSETTING CUSTOM DATASET STATE')
    #     return super().__setstate__(val)

    def __getitem__(self, idx):
        if 0:
            import multiprocessing
            print('\n\nidx = {!r}'.format(idx))
            print('self = {!r}'.format(self))
            print(multiprocessing.current_process())
        if self.storage_mode == 'ndsampler' or self.storage_mode == 'ndsampler-sql':
            sample = self.data.load_item(idx)
            data = sample['im'].ravel()[0:1].astype(np.float32)
            data_pt = torch.from_numpy(data)
        else:
            data = self.data[idx]
            data = np.array([data], dtype=np.int64)
            data_pt = torch.tensor(data)

        if self.return_mode == 'tensor':
            item = data_pt
        elif self.return_mode == 'dict':
            item = {
                'data': data_pt
            }
        elif self.return_mode == 'tuple':
            item = (data_pt,)
        elif self.return_mode == 'list':
            item = [data_pt]
        return item


def getsize(*objs):
    """
    sum size of object & members.
    https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python
    """
    import sys
    from types import ModuleType, FunctionType
    from gc import get_referents
    # Custom objects know their class.
    # Function objects seem to know way too much, including modules.
    # Exclude modules as well.
    blocklist = (type, ModuleType, FunctionType)
    # if isinstance(obj, blocklist):
    #     raise TypeError('getsize() does not take argument of type: ' + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = objs
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, blocklist) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size, len(seen_ids)


def byte_str(num, unit='auto', precision=2):
    """
    Automatically chooses relevant unit (KB, MB, or GB) for displaying some
    number of bytes.

    Args:
        num (int): number of bytes
        unit (str): which unit to use, can be auto, B, KB, MB, GB, TB, PB, EB,
            ZB, or YB.

    References:
        https://en.wikipedia.org/wiki/Orders_of_magnitude_(data)

    Returns:
        str: string representing the number of bytes with appropriate units

    Example:
        >>> num_list = [1, 100, 1024,  1048576, 1073741824, 1099511627776]
        >>> result = ub.repr2(list(map(byte_str, num_list)), nl=0)
        >>> print(result)
        ['0.00 KB', '0.10 KB', '1.00 KB', '1.00 MB', '1.00 GB', '1.00 TB']
    """
    abs_num = abs(num)
    if unit == 'auto':
        if abs_num < 2.0 ** 10:
            unit = 'KB'
        elif abs_num < 2.0 ** 20:
            unit = 'KB'
        elif abs_num < 2.0 ** 30:
            unit = 'MB'
        elif abs_num < 2.0 ** 40:
            unit = 'GB'
        elif abs_num < 2.0 ** 50:
            unit = 'TB'
        elif abs_num < 2.0 ** 60:
            unit = 'PB'
        elif abs_num < 2.0 ** 70:
            unit = 'EB'
        elif abs_num < 2.0 ** 80:
            unit = 'ZB'
        else:
            unit = 'YB'
    if unit.lower().startswith('b'):
        num_unit = num
    elif unit.lower().startswith('k'):
        num_unit =  num / (2.0 ** 10)
    elif unit.lower().startswith('m'):
        num_unit =  num / (2.0 ** 20)
    elif unit.lower().startswith('g'):
        num_unit = num / (2.0 ** 30)
    elif unit.lower().startswith('t'):
        num_unit = num / (2.0 ** 40)
    elif unit.lower().startswith('p'):
        num_unit = num / (2.0 ** 50)
    elif unit.lower().startswith('e'):
        num_unit = num / (2.0 ** 60)
    elif unit.lower().startswith('z'):
        num_unit = num / (2.0 ** 70)
    elif unit.lower().startswith('y'):
        num_unit = num / (2.0 ** 80)
    else:
        raise ValueError('unknown num={!r} unit={!r}'.format(num, unit))
    return ub.repr2(num_unit, precision=precision) + ' ' + unit


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    print('WORKER INIT FOR dataset')
    if hasattr(dataset.data, 'dset'):
        dset = dataset.data.dset
        if hasattr(dset, 'connect'):
            dset.connect(readonly=True)
        print('WORKER INIT FOR dset = {!r}'.format(dset))


def main(storage_mode='numpy', return_mode='tensor', total=24e5, shuffle=True, workers=2):
    """
    Args:
        storage_mode : how the dataset is stored in backend datasets

        return_mode : how each data item is returned

        total : size of backend storage

    """

    if 0:
        # torch_multiprocessing.get_context()
        torch.multiprocessing.set_start_method('spawn')

    mem = psutil.virtual_memory()
    start_mem = mem.used
    mem_str = byte_str(start_mem)
    print('Starting used system memory = {!r}'.format(mem_str))

    train_data = CustomDataset(
        storage_mode=storage_mode,
        return_mode=return_mode,
        total=total)

    if storage_mode == 'numpy':
        total_storate_bytes = train_data.data.dtype.itemsize * train_data.data.size
    else:
        total_storate_bytes = sys.getsizeof(train_data.data)
        # total_storate_bytes = getsize(self.data)
    print('total_storage_size = {!r}'.format(byte_str(total_storate_bytes)))

    mem = psutil.virtual_memory()
    mem_str = byte_str(mem.used - start_mem)
    print('After init CustomDataset   memory = {!r}'.format(mem_str))

    print('shuffle = {!r}'.format(shuffle))

    num_workers = workers
    batch_size = 32
    # batch_size = 300
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=shuffle, drop_last=True,
                              pin_memory=False, num_workers=num_workers,
                              worker_init_fn=worker_init_fn)

    used_nbytes = psutil.virtual_memory().used - start_mem
    print('After init DataLoader memory = {!r}'.format(byte_str(used_nbytes)))

    if True:
        # Estimate peak usage
        import gc
        all_obj_nbytes, num_objects = getsize(*gc.get_objects())
        python_ptr_size = int((np.log2(sys.maxsize) + 1) / 8)
        assert python_ptr_size == 8, 'should be 8 bytes on 64bit python'
        all_ptr_nbytes = (num_objects * python_ptr_size)

        prog_nbytes_estimated_1 = all_ptr_nbytes + all_obj_nbytes
        prog_nbytes_measured_2 = psutil.virtual_memory().used - start_mem
        print('prog_nbytes_estimated_1 = {!r}'.format(byte_str(prog_nbytes_estimated_1)))
        print('prog_nbytes_measured_2  = {!r}'.format(byte_str(prog_nbytes_measured_2)))

        peak_bytes_est1 = prog_nbytes_estimated_1 * (num_workers + 1)
        peak_bytes_est2 = prog_nbytes_measured_2 * (num_workers + 1)
        print('peak_bytes_est1 = {!r}'.format(byte_str(peak_bytes_est1)))
        print('peak_bytes_est2 = {!r}'.format(byte_str(peak_bytes_est2)))

    max_bytes = -float('inf')
    prog = ub.ProgIter(train_loader)
    for item in prog:
        used_bytes = psutil.virtual_memory().used - start_mem
        max_bytes = max(max_bytes, used_bytes)
        prog.set_extra(' Mem=' + byte_str(used_bytes))

    used_bytes = psutil.virtual_memory().used - start_mem
    print('measured final usage: {}'.format(byte_str(used_bytes)))
    print('measured peak usage:  {}'.format(byte_str(max_bytes)))

    if 0 and hasattr(train_data.data, 'frames'):
        sampler = train_data.data
        print('sampler.regions.__dict__ = {}'.format(
            ub.repr2(sampler.regions.__dict__, nl=1)))

        print('sampler.frames.__dict__ = {}'.format(
            ub.repr2(sampler.frames.__dict__, nl=1)))


if __name__ == '__main__':
    """

    CommandLine:
        python debug_memory.py numpy tensor --total=24e5 --shuffle=True

        cd ~/code/netharn/dev

        python debug_memory.py --storage_mode=numpy --total=24e5 --shuffle=True
        python debug_memory.py --storage_mode=numpy --total=24e5 --shuffle=False
        python debug_memory.py --storage_mode=python --total=24e5 --shuffle=True
        python debug_memory.py --storage_mode=python --total=24e5 --shuffle=False

        python debug_memory.py --storage_mode=ndsampler --total=100000 --shuffle=True --workers=4
        python debug_memory.py --storage_mode=ndsampler-sql --total=100000 --shuffle=True --workers=4

        python debug_memory.py --storage_mode=ndsampler --total=10000 --shuffle=True --workers=4
        python debug_memory.py --storage_mode=ndsampler-sql --total=10000 --shuffle=True --workers=4

        python debug_memory.py --storage_mode=ndsampler --total=1000 --shuffle=True --workers=0 --profile
        python debug_memory.py --storage_mode=ndsampler-sql --total=1000 --shuffle=True --workers=0 --profile

        python debug_memory.py --storage_mode=ndsampler --total=1000 --shuffle=False --workers=0

        python debug_memory.py --storage_mode=ndsampler --total=1000 --shuffle=False --workers=8
        python debug_memory.py --storage_mode=ndsampler-sql --total=1000 --shuffle=False --workers=8

        python debug_memory.py --storage_mode=ndsampler --total=1000 --shuffle=True --workers=0
        python debug_memory.py --storage_mode=ndsampler --total=1000 --shuffle=False --workers=0


        python debug_memory.py --storage_mode=ndsampler --total=1000 --shuffle=False --workers=4
        python debug_memory.py --storage_mode=ndsampler --total=1000 --shuffle=True --workers=4

        srun -c 5 -p community --gres=gpu:1 \
            python debug_memory.py --storage_mode=ndsampler --total=1000 --shuffle=True --workers=4

            python debug_memory.py --storage_mode=python --total=24e5 --shuffle=False --workers=4


        python debug_memory.py numpy dict 24e5
        python debug_memory.py python list 24e7

    Conclusions:

        * It seems like it is ok if the return type is a dictionary
          the problem seems to be localized to the storage type.
    """
    import fire
    fire.Fire(main)

"""

@VitalyFedyunin Let me see if I understand correctly, when you access an item
in a list you create a new reference to it, which will force its refcount to be
incremented (i.e. be written to).

pages are typically 4096 bytes.

"""


def test_manager():
    """
    Look at how managers works
    """
    from multiprocessing import Manager

    import kwcoco
    dset = kwcoco.CocoDataset.coerce('shapes32')

    manager = Manager()
    managed_imgs = manager.dict(dset.imgs)

    import timerit
    ti = timerit.Timerit(100, bestof=10, verbose=2)
    for timer in ti.reset('time'):
        with timer:
            managed_imgs.keys()

    for timer in ti.reset('time'):
        with timer:
            dset.imgs.keys()
