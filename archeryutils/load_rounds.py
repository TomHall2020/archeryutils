import json
from pathlib import Path
import warnings

from archeryutils.rounds import Pass, Round


def read_json_to_round_dict(json_filelist):
    """
    Subroutine to return round information read in from a json file as a dictionary of
    rounds

    Parameters
    ----------
    json_filelist : list of str
        filenames of json round files in ./round_data_files/

    Returns
    -------
    round_dict : dict of str : rounds.Round

    References
    ----------
    """
    if type(json_filelist) is not list:
        json_filelist = [json_filelist]

    round_data_files = Path(__file__).parent.joinpath("round_data_files")

    round_dict = {}

    for json_file in json_filelist:
        json_filepath = round_data_files.joinpath(json_file)
        with open(json_filepath) as json_round_file:
            data = json.load(json_round_file)

        for round_i in data:

            # Assign location
            if "location" not in round_i:
                warnings.warn(
                    f"No location provided for round {round_i['name']}. "
                    "Defaulting to None."
                )
                round_i["location"] = None
                round_i["indoor"] = False
            elif round_i["location"] in [
                "i",
                "I",
                "indoors",
                "indoor",
                "in",
                "inside",
                "Indoors",
                "Indoor",
                "In",
                "Inside",
            ]:
                round_i["indoor"] = True
                round_i["location"] = "indoor"
            elif round_i["location"] in [
                "o",
                "O",
                "outdoors",
                "outdoor",
                "out",
                "outside",
                "Outdoors",
                "Outdoor",
                "Out",
                "Outside",
            ]:
                round_i["indoor"] = False
                round_i["location"] = "outdoor"
            elif round_i["location"] in [
                "f",
                "F",
                "field",
                "Field",
                "woods",
                "Woods",
            ]:
                round_i["indoor"] = False
                round_i["location"] = "field"
            else:
                warnings.warn(
                    f"Location not recognised for round {round_i['name']}. "
                    "Defaulting to None"
                )
                round_i["indoor"] = False
                round_i["location"] = None

            # Assign governing body
            if "body" not in round_i:
                warnings.warn(
                    f"No body provided for round {round_i['name']}. "
                    "Defaulting to 'custom'."
                )
                round_i["body"] = "custom"
                # TODO: Could do sanitisation here e.g. AGB vs agb etc or trust user...

            # Assign round family
            if "family" not in round_i:
                warnings.warn(
                    f"No family provided for round {round_i['name']}. "
                    "Defaulting to ''."
                )
                round_i["family"] = ""

            # Assign passes
            passes = []
            for pass_i in round_i["passes"]:
                passes.append(
                    Pass(
                        pass_i["n_arrows"],
                        pass_i["diameter"] / 100,
                        pass_i["scoring"],
                        pass_i["distance"],
                        dist_unit=pass_i["dist_unit"],
                        indoor=round_i["indoor"],
                    )
                )

            round_dict[round_i["codename"]] = Round(
                round_i["name"],
                passes,
                location=round_i["location"],
                body=round_i["body"],
                family=round_i["family"],
            )

    return round_dict


class DotDict(dict):
    """
    A subclass of dict to provide dot notation access to a dictionary

    Attributes
    ----------

    Methods
    -------

    References
    -------
    https://goodcode.io/articles/python-dict-object/
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(self._attribute_err_msg(name))

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(self._attribute_err_msg(name))

    def _attribute_err_msg(self, name: str) -> str:
        quoted = [f"'{key}'" for key in self]
        return f"No such attribute: '{name}'. Please select from {', '.join(quoted)}."


# Generate a set of default rounds that come with this module, accessible as a DotDict:


def _make_rounds_dict(json_name: str) -> DotDict:
    return DotDict(read_json_to_round_dict(json_name))


AGB_outdoor_imperial = _make_rounds_dict("AGB_outdoor_imperial.json")
AGB_outdoor_metric = _make_rounds_dict("AGB_outdoor_metric.json")
AGB_indoor = _make_rounds_dict("AGB_indoor.json")
WA_outdoor = _make_rounds_dict("WA_outdoor.json")
WA_indoor = _make_rounds_dict("WA_indoor.json")
custom = _make_rounds_dict("Custom.json")

del _make_rounds_dict