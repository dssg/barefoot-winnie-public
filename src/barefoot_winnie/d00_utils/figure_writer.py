"""
``AbstractDataSet`` implementation to save matplotlib objects as image files.
"""

import os.path
from typing import Any, Dict, Optional

from kedro.io.core import AbstractDataSet, DataSetError, FilepathVersionMixIn


class MatplotlibWriter(AbstractDataSet, FilepathVersionMixIn):
    """
        ``MatplotlibWriter`` saves matplotlib objects as image files.

        Example:
        ::

            >>> import matplotlib.pyplot as plt
            >>> from kedro.contrib.io.matplotlib import MatplotlibWriter
            >>>
            >>> plt.plot([1,2,3],[4,5,6])
            >>>
            >>> single_plot_writer = MatplotlibWriter(filepath="docs/new_plot.png")
            >>> single_plot_writer.save(plt)
            >>>
            >>> plt.close()
            >>>
            >>> plots = dict()
            >>>
            >>> for colour in ['blue', 'green', 'red']:
            >>>     plots[colour] = plt.figure()
            >>>     plt.plot([1,2,3],[4,5,6], color=colour)
            >>>     plt.close()
            >>>
            >>> multi_plot_writer = MatplotlibWriter(filepath="docs/",
            >>>                                      save_args={'multiFile': True})
            >>> multi_plot_writer.save(plots)

    """

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            load_args=self._load_args,
            save_args=self._save_args,
        )

    def __init__(
        self,
        filepath: str,
        load_args: Optional[Dict[str, Any]] = None,
        save_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Creates a new instance of ``MatplotlibWriter``.

        Args:
            filepath: path to a text file.
            load_args: Currently ignored as loading is not supported.
            save_args: multiFile: allows for multiple plot objects
                to be saved. Additional load arguments can be found at
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
        """
        default_save_args = {"multiFile": True}
        default_load_args = {}

        self._filepath = filepath
        self._load_args = self._handle_default_args(load_args, default_load_args)
        self._save_args = self._handle_default_args(save_args, default_save_args)
        self._mutlifile_mode = self._save_args.get("multiFile")
        self._save_args.pop("multiFile")

    @staticmethod
    def _handle_default_args(user_args: dict, default_args: dict) -> dict:
        return {**default_args, **user_args} if user_args else default_args

    def _load(self) -> str:
        raise DataSetError("Loading not supported for MatplotlibWriter")

    def _save(self, data) -> None:

        if self._mutlifile_mode:

            if not os.path.isdir(self._filepath):
                os.makedirs(self._filepath)

            if isinstance(data, list):
                for index, plot in enumerate(data):
                    plot.savefig(
                        os.path.join(self._filepath, str(index) + '.png'), **self._save_args
                    )

            elif isinstance(data, dict):
                for plot_name, plot in data.items():
                    plot.savefig(
                        os.path.join(self._filepath, plot_name + '.png'), **self._save_args
                    )

            else:
                plot_type = type(data)
                raise DataSetError(
                    (
                        "multiFile is True but data type "
                        "not dict or list. Rather, {}".format(plot_type)
                    )
                )

        else:
            data.savefig(self._filepath, **self._save_args)

    def _exists(self) -> bool:
        return os.path.isfile(self._filepath)
