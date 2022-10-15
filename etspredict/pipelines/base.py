#!/usr/bin/env python3

from .ioutils import IOManager


class Pipeline(IOManager):
    def __init__(
        self,
        timeseries,
        root_dir,
        path_elements,
        file_parameters,
        save_index=True,
        *args,
        **kwargs,
    ):
        """Constructions output directory according to given parameters."""

        super().__init__(
            root_dir=root_dir,
            path_elements=path_elements,
            file_parameters=file_parameters,
            save_index=save_index,
            *args,
            **kwargs,
        )
        self.construct_output_directory()
        print(f"Output will be saved at {self.output_directory}!")
        self.timeseries = timeseries

    def run_wrapped_function(self, func, *args, **kwargs):
        """Takes a function that must take in the timeseries as np.array and
        return pd.DataFrame and runs function, saves result at location
        defined by IOManager

        """

        result = func(self.timeseries, *args, **kwargs)
        self.save_dataframe(result)

        return result
