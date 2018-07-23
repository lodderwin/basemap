import os


class Utils():
    @staticmethod
    def curr_path():
        """Return path this file resides in."""
        return os.path.dirname(os.path.abspath(__file__))
