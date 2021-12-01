"""
Python implementation of gamma cascade ()

Nicholas Carrara [nmcarrara@ucdavis.edu]
Luca Pagani      [lpagani@ucdavis.edu]
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List

@dataclass
class Gamma:
    """A class used to represent a single gamma

    Attributes
    ----------
    energy: float
        A float for this gammas energy
    rel_intensity: float
        The relative intensity of this gamma with respect to its level
    tot_intensity: float
        The total intensity
    conv_coefficient: float
        Conversion coefficient for this gamma

    """
    energy: float
    rel_intensity: float
    tot_intensity: float
    conv_coefficient: float

@dataclass
class Level:
    """A class representing a level in a gamma cascade model
    
    Attributes
    ----------
    energy: float
        Energy for this level
    spin_parity: str
        The spin parity for this level
    half_life: str
        The half-life for this level
    gammas: List[Gamma]
        The list of gammas that are emitted from this level
    probabilities: List[float]
        The probabilities for each gamma
    cumulative_probabilities: List[float]
        The cumulative probability for each gamma
    next_level: List[int]
        
    """
    energy: float
    spin_parity: str
    half_life: str
    gammas: list = field(default_factory=list)
    probabilities: list = field(default_factory=list)
    cumulative_probabilities: list = field(default_factory=list)
    next_level: list = field(default_factory=list)

    def add_gamma(self,
        energy, rel_intensity, tot_intensity, conv_coefficient
    ) -> None:
        """Method to add a new gamma to the list"""
        self.gammas.append(
            Gamma(energy,rel_intensity,tot_intensity,conv_coefficient)
        )

    def construct_probabilities(self) -> None:
        """Method to construct probabilities and 
        cumulative probabilities for the list of gammas
        
        """
        level_sum = sum([gamma.rel_intensity for gamma in self.gammas])
        for idx, gamma in enumerate(self.gammas):
            self.probabilities.append(gamma.rel_intensity/level_sum)
            if idx == 0:
                self.cumulative_probabilities.append(self.probabilities[0])
            else:
                self.cumulative_probabilities.append(
                    self.probabilities[idx] + self.cumulative_probabilities[idx-1])


class ENDFParser:
    """Class for parsing ENDF txt files and generating gamma cascade models.
    
    """
    def __init__(self,
        input_file: str,
    ) -> None:
        """Init function for ENDF parser.
        
        """
        self.input_file = input_file
        self.levels = []
        self.parse()
        self.construct_probabilities()
        self.construct_next_levels()
    
    def parse(self) -> None:
        """Function for parsing ENDF files"""
        with open(self.input_file, "r") as file:
            # variables needed for parsing
            rel_photon_intensity = ""
            conv_coefficient = ""
            level_energy = ""
            gamma_energy = ""
            half_life = ""
            total_transition_intensity = ""
            spin_parity = ""
            for row in file:
                if len(gamma_energy) != 0:
                    lev_energy = float(level_energy)
                    found_match = False
                    for l in range(len(self.levels)):
                        if self.levels[l].energy == lev_energy:
                            found_match = True
                            break
                    if found_match:
                        if rel_photon_intensity.isspace():
                            rel_photon_intensity = 0.
                        self.levels[l].add_gamma(
                            float(gamma_energy),
                            float(rel_photon_intensity),
                            0,
                            float(conv_coefficient))
                    else:
                        self.levels.append(
                            Level(
                                float(level_energy),
                                spin_parity,
                                half_life)
                        )
                        if rel_photon_intensity.isspace():
                            rel_photon_intensity = 0.
                        self.levels[-1].add_gamma(
                            float(gamma_energy),
                            float(rel_photon_intensity),
                            0,
                            float(conv_coefficient)
                        )
                    gamma_energy = ""
                else:
                    if (len(row) >= 61):
                        if (row[5] == ' ' and row[6] == ' ' and row[7] == 'L'):
                            level_energy = row[9:17]
                            spin_parity = row[21:37]
                            half_life = row[39:47]
                        if (row[5] == ' ' and row[6] == ' ' and row[7] == 'G'):
                            gamma_energy = row[9:17]
                            rel_photon_intensity = row[21:27]
                            total_transition_intensity = row[64:72]

                            if (row[55] == ' '):
                                conv_coefficient = '00000000'
                            else:
                                conv_coefficient = row[55:60]

    def __getitem__(self, idx):
        """Overload of [] operator"""
        if idx < len(self.levels):
            return self.levels[idx]
        else:
            return self.levels[0]
    
    def max(self):
        """Returns the last level"""
        return self.levels[-1]
    
    def min(self):
        """Returns the first level"""
        return self.levels[0]
    
    def construct_probabilities(self):
        """Construct probabilities for each level"""
        for level in self.levels:
            level.construct_probabilities()
    
    def construct_next_levels(self):
        """Construct the next levels for each level"""
        for l_idx in range(len(self.levels)):
            for g_idx in range(len(self.levels[l_idx].gammas)):
                match_idx = 0
                match_diff = 10e10
                for t_idx in range(len(self.levels)):
                    temp_diff = np.fabs(
                        self.levels[l_idx].energy - 
                        self.levels[l_idx].gammas[g_idx].energy - 
                        self.levels[t_idx].energy)
                    if temp_diff < match_diff:
                        match_idx = t_idx
                        match_diff = temp_diff
                self.levels[l_idx].next_level.append(match_idx)
    
    def print(self):
        """Print out the levels and gamma information"""
        for idx, level in enumerate(self.levels):
            print("Level {} - {} MeV".format(idx, level.energy))
            for jdx, gamma in enumerate(level.gammas):
                output =  f"  - {jdx}) "
                output += f"{gamma.energy} --> level "
                output += f"{level.next_level[jdx]} - p(l-"
                output += f"{level.next_level[jdx]}|g-{jdx}) = "
                output += f"{level.probabilities[jdx]}"
                print(output)


if __name__ == "__main__":
    # try import cascade
    input_file = "../../data/ENDF/ar36_neutron_capture.txt"
    parser = ENDFParser(input_file)
    parser.print()



