import torch
import ubelt as ub


class RandomDataset(torch.utils.data.Dataset):
    """
    A torch dataset

    Simple black-on-white and white-on-black images.

    Example:
        RandomDataset(modes=5)[0]

    """
    def __init__(self, num=10, shape=(3, 32, 32), modes=None):
        self.num = num
        self.shape = shape
        self.modes = modes

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        cidx = torch.randint(0, 10, (1,))
        item = {}
        item['label'] = cidx
        if self.modes is None:
            item['inputs'] = torch.rand(*self.shape)
        else:
            modes_lut = {}
            for mode_x in range(self.modes):
                modes_lut['mode_' + str(mode_x)] = torch.rand(*self.shape)
            item['inputs'] = modes_lut
        return item


def nested_move(batch, device):
    walker = ub.IndexableWalker(batch)
    for path, val in walker:
        if isinstance(val, (list, dict)):
            continue
        elif isinstance(val, torch.Tensor):
            walker[path] = val.to(device)
        else:
            raise TypeError(path)
    return batch


def walk_tensors(nested):
    walker = ub.IndexableWalker(nested)
    for path, val in walker:
        if isinstance(val, torch.Tensor):
            yield path, val


def stub_model_forward(model=None, batch=None):
    parts = []
    for path, val in walk_tensors(batch):
        # Do some work
        flat = val.ravel()
        parts.append(flat)

    dummy_input = torch.cat(parts).float()[None, :]
    if model is not None:
        outputs = model(dummy_input)
    else:
        outputs = dummy_input[0:2] + dummy_input.sum()
    return outputs


def exhaust_loader(loader, device, model=None):
    # device = torch.device('cpu')
    _iter = iter(loader)
    # _iter = ub.ProgIter(_iter, total=len(loader), desc=key)
    # iter(loader)
    for batch in _iter:
        batch = nested_move(batch, device)

        outputs = stub_model_forward(model=model, batch=batch)
        outputs.data.cpu()  # NOQA

        # if device
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elif device.type == 'cpu':
            pass
        else:
            raise TypeError(device.type)


def batch_storage_size(batch):
    # Do some operation with all the data
    # layer_sums = {}
    import kwarray
    item_bytes = {}
    for path, val in walk_tensors(batch):
        # Do some work
        info = kwarray.dtype_info(val.dtype)
        num_items = val.numel()
        size_bytes = (num_items * info.bits) // 8
        item_bytes[tuple(path)] = size_bytes
    total_bytes = sum(item_bytes.values())
    import pint
    ureg = pint.UnitRegistry()
    total = total_bytes * ureg.byte
    megabytes = total.to('megabyte')
    # print('megabytes = {!r}'.format(megabytes))
    return megabytes


def main():
    """
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/netharn/dev/bench'))
    from bench_collate import *  # NOQA
    """
    import ubelt as ub
    import timerit

    param_basis = {
        'collate': ['default', 'identity'],
        'device': [
            'cpu', 1],
        # 'num': [10, 100],
        # 'num': [300, 1000, 10000],
        'num': [128],
        'shape': [
            (3, 224, 224),
            (2, 7, 224, 224),
            # (5, 13, 224, 224),
        ],
        # 'batch_size': [8, 64, 128],
        # 'modes': [None, 8, 128],
        'batch_size': [8],
        'modes': [None, 8],
    }

    grid = list(ub.named_product(param_basis))
    print('grid = {}'.format(ub.repr2(grid, nl=1)))

    collate_lut = {
        'identity': ub.identity,
        'default': torch.utils.data.dataloader.default_collate,
    }

    rows = []
    ti = timerit.Timerit(1, bestof=1, verbose=1)
    for params in ub.ProgIter(grid, verbose=3):
        config = params  # NOQA
        func = RandomDataset.__init__  # NOQA
        key = ub.repr2(params, nobr=1, nl=0, sep='', explicit=True, itemsep='')

        rows.append(params)
        param_subsets = {
            'dataset': ub.compatible(params, RandomDataset.__init__, start=1),
            'dataloader': ub.compatible(params, torch.utils.data.DataLoader.__init__, start=1),
            'other': ub.dict_isect(params, {'collate', 'device'})
        }
        print('param_subsets = {}'.format(ub.repr2(param_subsets, nl=1)))
        used = set.union(*map(set, param_subsets.values()))
        unknown = ub.dict_diff(params, used)
        assert len(unknown) == 0

        other_config = param_subsets['other']

        dataset = RandomDataset(**param_subsets['dataset'])

        loader = torch.utils.data.DataLoader(
            dataset, collate_fn=collate_lut[other_config['collate']],
            **param_subsets['dataloader'])

        batch = ub.peek(loader)
        total_bytes = batch_storage_size(batch)
        params[str(total_bytes.units)] = total_bytes.magnitude

        device = torch.device(other_config['device'])
        model = torch.nn.LazyLinear(13)
        model = model.to(device)
        for timer in ti.reset(key):
            with timer:
                exhaust_loader(loader, device, model)

        params['min_time'] = ti.min()
        params['key'] = key

    rankings = ti.rankings['min']
    print('rankings = {}'.format(ub.repr2(rankings, nl=2, align=':', precision=6)))

    import pandas as pd
    df = pd.DataFrame(rows)
    df = df.sort_values('min_time')
    print(df)

    keys = ['device', 'collate']
    agg_stats = {}
    import kwarray
    for key in keys:
        agg = {}
        for subkey, subdf in df.groupby(key):
            agg[subkey] = ub.dict_isect(kwarray.stats_dict(subdf['min_time']), {'mean', 'std', 'max'})
        agg_stats[key] = agg

    print('agg_stats = {}'.format(ub.repr2(agg_stats, nl=-1)))
    import os
    _draw_stats(df)
    if ub.argflag('--show') or os.environ.get('BENCH_PLOTS', ''):
        _draw_stats(df)

    return df


def _draw_stats(df):
    import kwplot
    plt = kwplot.autoplt()
    import seaborn as sns
    sns.set()

    fig = kwplot.figure(fnum=1, doclf=True)
    ax = fig.gca()
    # sns.scatterplot(data=df, x='key', y='min_time', hue='collate', ax=ax)
    sns.scatterplot(data=df, x='megabyte', y='min_time', hue='collate', ax=ax)
    [tick.set_rotation(75) for tick in ax.get_xticklabels()]
    fig.subplots_adjust(bottom=0.3)
    ax.set_title(
        'Benchmarks for collate functions \n'
        '(It looks like identity collate is not significantly worse)')

    fig = kwplot.figure(fnum=2, doclf=True)
    ax = fig.gca()
    sns.violinplot(data=df, x='collate', y='min_time', hue='collate', ax=ax)
    [tick.set_rotation(75) for tick in ax.get_xticklabels()]

    fig.subplots_adjust(bottom=0.18, left=0.2)
    ax.set_title(
        'Benchmarks for collate functions \n'
        '(It looks like identity collate is not significantly worse)')

    plt.show()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/dev/bench/bench_collate.py --show
        python ~/code/netharn/dev/bench/bench_collate.py --show
    """
    main()
