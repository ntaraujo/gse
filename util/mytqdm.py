from datetime import datetime, timedelta
from tqdm import tqdm
from tqdm.std import Bar
from tqdm.utils import _unicode, _is_ascii, FormatReplace, disp_len, disp_trim
from proglog import TqdmProgressBarLogger




class MyLogger(TqdmProgressBarLogger):

    def __init__(self, my_callback, *args, **kwargs):
        global mytqdm_callback
        mytqdm_callback = my_callback
        super().__init__(*args, **kwargs)
        self.tqdm = mytqdm
mytqdm_callback = lambda s, f: None


class mytqdm(tqdm):

    def mytqdm_callback(self, format_dict):
        global mytqdm_callback
        mytqdm_callback(self, format_dict)

    def format_meter(self, n, total, elapsed, ncols=None, prefix='', ascii=False,
                     unit='it', unit_scale=False, rate=None, bar_format=None,
                     postfix=None, unit_divisor=1000, initial=0, colour=None,
                     **extra_kwargs):
        """
        Return a string-based progress bar given some parameters

        Parameters
        ----------
        n  : int or float
            Number of finished iterations.
        total  : int or float
            The expected total number of iterations. If meaningless (None),
            only basic progress statistics are displayed (no ETA).
        elapsed  : float
            Number of seconds passed since start.
        ncols  : int, optional
            The width of the entire output message. If specified,
            dynamically resizes `{bar}` to stay within this bound
            [default: None]. If `0`, will not print any bar (only stats).
            The fallback is `{bar:10}`.
        prefix  : str, optional
            Prefix message (included in total width) [default: ''].
            Use as {desc} in bar_format string.
        ascii  : bool, optional or str, optional
            If not set, use unicode (smooth blocks) to fill the meter
            [default: False]. The fallback is to use ASCII characters
            " 123456789#".
        unit  : str, optional
            The iteration unit [default: 'it'].
        unit_scale  : bool or int or float, optional
            If 1 or True, the number of iterations will be printed with an
            appropriate SI metric prefix (k = 10^3, M = 10^6, etc.)
            [default: False]. If any other non-zero number, will scale
            `total` and `n`.
        rate  : float, optional
            Manual override for iteration rate.
            If [default: None], uses n/elapsed.
        bar_format  : str, optional
            Specify a custom bar string formatting. May impact performance.
            [default: '{l_bar}{bar}{r_bar}'], where
            l_bar='{desc}: {percentage:3.0f}%|' and
            r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, '
              '{rate_fmt}{postfix}]'
            Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
              percentage, elapsed, elapsed_s, ncols, nrows, desc, unit,
              rate, rate_fmt, rate_noinv, rate_noinv_fmt,
              rate_inv, rate_inv_fmt, postfix, unit_divisor,
              remaining, remaining_s, eta.
            Note that a trailing ": " is automatically removed after {desc}
            if the latter is empty.
        postfix  : *, optional
            Similar to `prefix`, but placed at the end
            (e.g. for additional stats).
            Note: postfix is usually a string (not a dict) for this method,
            and will if possible be set to postfix = ', ' + postfix.
            However other types are supported (#382).
        unit_divisor  : float, optional
            [default: 1000], ignored unless `unit_scale` is True.
        initial  : int or float, optional
            The initial counter value [default: 0].
        colour  : str, optional
            Bar colour (e.g. 'green', '#00ff00').

        Returns
        -------
        out  : Formatted meter and stats, ready to display.
        """

        # sanity check: total
        if total and n >= (total + 0.5):  # allow float imprecision (#849)
            total = None

        # apply custom scale if necessary
        if unit_scale and unit_scale not in (True, 1):
            if total:
                total *= unit_scale
            n *= unit_scale
            if rate:
                rate *= unit_scale  # by default rate = 1 / self.avg_time
            unit_scale = False

        elapsed_str = self.format_interval(int(elapsed))

        # if unspecified, attempt to use rate = average speed
        # (we allow manual override since predicting time is an arcane art)
        if rate is None and elapsed:
            rate = (n - initial) / elapsed
        inv_rate = 1 / rate if rate else None
        format_sizeof = self.format_sizeof
        rate_noinv_fmt = ((format_sizeof(rate) if unit_scale else
                           '{0:5.2f}'.format(rate))
                          if rate else '?') + unit + '/s'
        rate_inv_fmt = ((format_sizeof(inv_rate) if unit_scale else
                         '{0:5.2f}'.format(inv_rate))
                        if inv_rate else '?') + 's/' + unit
        rate_fmt = rate_inv_fmt if inv_rate and inv_rate > 1 else rate_noinv_fmt

        if unit_scale:
            n_fmt = format_sizeof(n, divisor=unit_divisor)
            total_fmt = format_sizeof(total, divisor=unit_divisor) \
                if total is not None else '?'
        else:
            n_fmt = str(n)
            total_fmt = str(total) if total is not None else '?'

        try:
            postfix = ', ' + postfix if postfix else ''
        except TypeError:
            pass

        remaining = (total - n) / rate if rate and total else 0
        remaining_str = self.format_interval(remaining) if rate else '?'
        try:
            eta_dt = datetime.now() + timedelta(seconds=remaining) \
                if rate and total else datetime.utcfromtimestamp(0)
        except OverflowError:
            eta_dt = datetime.max

        # format the stats displayed to the left and right sides of the bar
        if prefix:
            # old prefix setup work around
            bool_prefix_colon_already = (prefix[-2:] == ": ")
            l_bar = prefix if bool_prefix_colon_already else prefix + ": "
        else:
            l_bar = ''

        r_bar = '| {0}/{1} [{2}<{3}, {4}{5}]'.format(
            n_fmt, total_fmt, elapsed_str, remaining_str, rate_fmt, postfix)

        # Custom bar formatting
        # Populate a dict with all available progress indicators
        format_dict = dict(
            # slight extension of self.format_dict
            n=n, n_fmt=n_fmt, total=total, total_fmt=total_fmt,
            elapsed=elapsed_str, elapsed_s=elapsed,
            ncols=ncols, desc=prefix or '', unit=unit,
            rate=inv_rate if inv_rate and inv_rate > 1 else rate,
            rate_fmt=rate_fmt, rate_noinv=rate,
            rate_noinv_fmt=rate_noinv_fmt, rate_inv=inv_rate,
            rate_inv_fmt=rate_inv_fmt,
            postfix=postfix, unit_divisor=unit_divisor,
            colour=colour,
            # plus more useful definitions
            remaining=remaining_str, remaining_s=remaining,
            l_bar=l_bar, r_bar=r_bar, eta=eta_dt,
            **extra_kwargs)

        # total is known: we can predict some stats
        if total:
            # fractional and percentage progress
            frac = n / total
            percentage = frac * 100

            l_bar += '{0:3.0f}%|'.format(percentage)

            if ncols == 0:
                return l_bar[:-1] + r_bar[1:]

            format_dict.update(l_bar=l_bar)
            if bar_format:
                format_dict.update(percentage=percentage)

                # auto-remove colon for empty `desc`
                if not prefix:
                    bar_format = bar_format.replace("{desc}: ", '')
            else:
                bar_format = "{l_bar}{bar}{r_bar}"
            self.mytqdm_callback(format_dict)

            full_bar = FormatReplace()
            try:
                nobar = bar_format.format(bar=full_bar, **format_dict)
            except UnicodeEncodeError:
                bar_format = _unicode(bar_format)
                nobar = bar_format.format(bar=full_bar, **format_dict)
            if not full_bar.format_called:
                # no {bar}, we can just format and return
                return nobar

            # Formatting progress bar space available for bar's display
            full_bar = Bar(
                frac,
                max(1, ncols - disp_len(nobar))
                if ncols else 10,
                charset=Bar.ASCII if ascii is True else ascii or Bar.UTF,
                colour=colour)
            if not _is_ascii(full_bar.charset) and _is_ascii(bar_format):
                bar_format = _unicode(bar_format)
            res = bar_format.format(bar=full_bar, **format_dict)
            return disp_trim(res, ncols) if ncols else res

        elif bar_format:
            # user-specified bar_format but no total
            l_bar += '|'
            format_dict.update(l_bar=l_bar, percentage=0)
            self.mytqdm_callback(format_dict)
            full_bar = FormatReplace()
            nobar = bar_format.format(bar=full_bar, **format_dict)
            if not full_bar.format_called:
                return nobar
            full_bar = Bar(
                0,
                max(1, ncols - disp_len(nobar))
                if ncols else 10,
                charset=Bar.BLANK,
                colour=colour)
            res = bar_format.format(bar=full_bar, **format_dict)
            return disp_trim(res, ncols) if ncols else res
        else:
            # no total: no progressbar, ETA, just progress stats
            return ((prefix + ": ") if prefix else '') + \
                '{0}{1} [{2}, {3}{4}]'.format(
                    n_fmt, unit, elapsed_str, rate_fmt, postfix)
