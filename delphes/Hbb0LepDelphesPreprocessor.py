from delphes.DelphesPreprocessor import DelphesPreprocessor
import numpy as np

class Hbb0LepDelphesPreprocessor(DelphesPreprocessor):

    def __init__(self):
        # the original branches needed for all subsequent processing steps
        self.input_branches = ["Event.Weight", "Electron.PT", "Muon.PT", "Jet.PT", "Jet.Mass", "Jet.Flavor", "Jet.Phi", "Jet.Eta", "MissingET.MET", "MissingET.Phi"]

        # the branches that will finally be exported
        self.permanent_output_branches = ["EventWeight", "MET", "pTB1", "pTB2", "mBB", "dRBB", "dEtaBB", "dPhiMETdijet", "SumPtJet", "nJ"]
        #self.debugging_output_branches = ["Electron.PT", "Muon.PT", "number_hard_leptons", "MissingET.MET", "number_b_jets_truth_tagging", "Jet.Flavor", "H_T", "Jet.PT", "b_jet_pt"]
        self.output_branches = self.permanent_output_branches# + self.debugging_output_branches

    def load(self, infile_path):
        super(Hbb0LepDelphesPreprocessor, self).load(infile_path = infile_path, branches = self.input_branches)

    def process(self, lumi, xsec):
        print("running with lumi = {} fb^-1".format(lumi))
        print("running with xsec = {} pb".format(xsec))

        # get the sum-of-weights of the loaded events
        sow = float(sum(self._extract_column("Event.Weight")))
        print("found SOW = {}".format(sow))

        weight_modifier = lumi * xsec * 1000 / sow # the factor of 1000 converts between fb and pb

        # ensure the correct normalization of these events
        self._add_column("EventWeight", lambda row: row["Event.Weight"][0] * weight_modifier)

        # count the number of electrons and muons (== leptons) above 7 GeV
        self._add_column("number_hard_muons", lambda row: sum(row["Muon.PT"] > 7.0))
        self._add_column("number_hard_electrons", lambda row: sum(row["Electron.PT"] > 7.0))
        self._add_column("number_hard_leptons", lambda row: row["number_hard_muons"] + row["number_hard_electrons"])

        # select only events with no hard leptons
        self._select(lambda row: row["number_hard_leptons"] == 0)

        # select only events with MET > 150 GeV
        self._add_column("MET", lambda row: row["MissingET.MET"][0]) # Delphes stores MET as a vector of length 1
        self._select(lambda row: row["MET"] > 150)

        # select only events with exactly 2 hard b-jets (use truth-tagging here to increase statistics)
        self._add_column("number_b_jets_truth_tagging", lambda row: sum(row["Jet.Flavor"] == 5))
        self._add_column("number_hard_b_jets", lambda row: sum(np.logical_and(row["Jet.PT"] > 45.0, row["Jet.Flavor"] == 5)))
        self._select(lambda row: row["number_b_jets_truth_tagging"] == 2 and row["number_hard_b_jets"] >= 1)

        # also, use just events with either 2 or 3 jets
        self._add_column("number_jets", lambda row: len(row["Jet.Flavor"]))
        self._add_column("nJ", lambda row: row["number_jets"])
        self._select(lambda row: row["number_jets"] == 2 or row["number_jets"] == 3)

        # cut on the scalar sum of jet p_T
        self._add_column("SumPtJet", lambda row: sum(row["Jet.PT"]))
        def selection_SumPtJet(row):
            if row["number_jets"] == 2:
                return row["SumPtJet"] > 120
            elif row["number_jets"] == 3:
                return row["SumPtJet"] > 150
            else:
                raise Exception("Error: this should not happen!")
        self._select(selection_SumPtJet)

        # at this point, can flatten the properties of the two b-jets
        self._add_column("b_jet_pt", lambda row: row["Jet.PT"][row["Jet.Flavor"] == 5])
        self._add_column("b_jet_phi", lambda row: row["Jet.Phi"][row["Jet.Flavor"] == 5])
        self._add_column("b_jet_eta", lambda row: row["Jet.Eta"][row["Jet.Flavor"] == 5])
        self._add_column("b_jet_m", lambda row: row["Jet.Mass"][row["Jet.Flavor"] == 5])
        self._add_column("pTB1", lambda row: row["b_jet_pt"][0])
        self._add_column("pTB2", lambda row: row["b_jet_pt"][1])
        self._add_column("phiB1", lambda row: row["b_jet_phi"][0])
        self._add_column("phiB2", lambda row: row["b_jet_phi"][1])
        self._add_column("etaB1", lambda row: row["b_jet_eta"][0])
        self._add_column("etaB2", lambda row: row["b_jet_eta"][1])
        self._add_column("mB1", lambda row: row["b_jet_m"][0])
        self._add_column("mB2", lambda row: row["b_jet_m"][1])

        # also compute the jet energies of the two b-jets (needed later to get mBB)
        self._add_column("EB1", lambda row: np.sqrt(row["mB1"] ** 2 + (row["pTB1"] * np.cosh(row["etaB1"])) ** 2))
        self._add_column("pxB1", lambda row: row["pTB1"] * np.cos(row["phiB1"]))
        self._add_column("pyB1", lambda row: row["pTB1"] * np.sin(row["phiB1"]))
        self._add_column("pzB1", lambda row: row["pTB1"] * np.sinh(row["etaB1"]))

        self._add_column("EB2", lambda row: np.sqrt(row["mB2"] ** 2 + (row["pTB2"] * np.cosh(row["etaB2"])) ** 2))
        self._add_column("pxB2", lambda row: row["pTB2"] * np.cos(row["phiB2"]))
        self._add_column("pyB2", lambda row: row["pTB2"] * np.sin(row["phiB2"]))
        self._add_column("pzB2", lambda row: row["pTB2"] * np.sinh(row["etaB2"]))

        # get the Higgs candidate
        self._add_column("pxBB", lambda row: row["pxB1"] + row["pxB2"])
        self._add_column("pyBB", lambda row: row["pyB1"] + row["pyB2"])
        self._add_column("pzBB", lambda row: row["pzB1"] + row["pzB2"])
        self._add_column("EBB", lambda row: row["EB1"] + row["EB2"])
        self._add_column("phiBB", lambda row: np.arctan2(row["pyBB"], row["pxBB"]))

        # make sure that the MET does not point in the direction of any of the jets
        def delta_phi(phi1, phi2):
            dphi = abs(phi1 - phi2)
            if dphi > np.pi:
                dphi = 2 * np.pi - dphi

            return dphi

        # compute some of its properties
        self._add_column("dEtaBB", lambda row: abs(row["etaB1"] - row["etaB2"]))
        self._add_column("dRBB", lambda row: np.sqrt((row["etaB1"] - row["etaB2"]) ** 2 + delta_phi(row["phiB1"], row["phiB2"]) ** 2))

        def MET_jet_OR(row):
            if row["number_jets"] == 3:
                dphi = np.pi / 6
            elif row["number_jets"] == 2:
                dphi = np.pi / 6
            else:
                raise Exception("Error: this should not happen!")
                
            jets_alive = [delta_phi(row["MissingET.Phi"][0], jet_phi) > dphi for jet_phi in row["Jet.Phi"]]
            return all(jets_alive)

        self._select(MET_jet_OR)

        # make sure the MET and the Higgs candidate are separated
        self._add_column("dPhiMETdijet", lambda row: delta_phi(row["phiBB"], row["MissingET.Phi"][0]))
        self._select(lambda row: row["dPhiMETdijet"] > 2 * np.pi / 3)

        # make sure the two b-jets are not too close to each other
        self._select(lambda row: delta_phi(row["phiB1"], row["phiB2"]) < 7 * np.pi / 9)

        # compute the invariant mass of the two b-jets
        self._add_column("mBB", lambda row: np.sqrt(row["EBB"] ** 2 - row["pxBB"] ** 2 - row["pyBB"] ** 2 - row["pzBB"] ** 2))
                         
        return self.df[self.output_branches]
