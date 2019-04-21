from delphes.DelphesPreprocessor import DelphesPreprocessor
import numpy as np

class Hbb1LepDelphesPreprocessor(DelphesPreprocessor):

    def __init__(self):
        self.input_branches = ["Event.Weight", "Electron.PT", "Electron.Eta", "Electron.Phi", "Muon.PT", "Muon.Eta", "Muon.Phi", "Jet.PT", "Jet.Mass", "Jet.Flavor", "Jet.Phi", "Jet.Eta", "MissingET.MET", "MissingET.Phi"]

        self.permanent_output_branches = ["EventWeight", "MET", "pTB1", "pTB2", "mBB", "dRBB", "dEtaBB", "dPhiVH", "dPhilb", "mTW", "mtop", "dEtaVH", "pTV"]
        self.output_branches = self.permanent_output_branches

    def load(self, infile_path):
        super(Hbb1LepDelphesPreprocessor, self).load(infile_path = infile_path, branches = self.input_branches)

    def process(self, lumiweight):
        if self.df is None:
            return None

        if len(self.df) <= 1:
            return None
            
        # ensure the correct normalization of these events
        self._add_column("EventWeight", lambda row: row["Event.Weight"][0] * lumiweight)

        # count the number of electrons and muons (== leptons) above 7 GeV
        self._add_column("number_hard_muons", lambda row: sum(row["Muon.PT"] > 25.0))
        self._add_column("number_hard_electrons", lambda row: sum(row["Electron.PT"] > 27.0))
        self._add_column("number_hard_leptons", lambda row: row["number_hard_muons"] + row["number_hard_electrons"])

        # select only events with exactly one hard lepton
        self._select(lambda row: row["number_hard_leptons"] == 1)

        # select only events with either 2 or 3 jets passing some pT requirement
        self._add_column("number_passing_jets", lambda row: sum(row["Jet.PT"][row["Jet.Eta"] < 2.5] > 20) + sum(row["Jet.PT"][row["Jet.Eta"] > 2.5] > 30))
        self._select(lambda row: row["number_passing_jets"] == 2 or row["number_passing_jets"] == 3)

        # select only events with exactly 2 hard b-jets (use truth-tagging here to increase statistics)
        self._add_column("number_b_jets_truth_tagging", lambda row: sum(row["Jet.Flavor"] == 5))
        self._add_column("number_hard_b_jets", lambda row: sum(np.logical_and(row["Jet.PT"] > 45.0, row["Jet.Flavor"] == 5)))
        self._select(lambda row: row["number_b_jets_truth_tagging"] == 2 and row["number_hard_b_jets"] >= 1)

        # put selection on the MET
        self._add_column("MET", lambda row: row["MissingET.MET"][0]) # Delphes stores MET as a vector of length 1
        def selection_MET(row):
            if row["number_hard_electrons"] == 1:
                return row["MET"] > 30
            elif row["number_hard_muons"] == 1:
                return True # no additional MET requirement for muons
            else:
                raise Exception("Error: this should not happen!")
        self._select(selection_MET)

        # flatten the properties of the single lepton in the event (take it as massless)
        self._add_column("lepton_pt", lambda row: np.concatenate([row["Muon.PT"][row["Muon.PT"] > 25.0], row["Electron.PT"][row["Electron.PT"] > 27]])[0])
        self._add_column("lepton_eta", lambda row: np.concatenate([row["Muon.Eta"][row["Muon.PT"] > 25.0], row["Electron.Eta"][row["Electron.PT"] > 27]])[0])
        self._add_column("lepton_phi", lambda row: np.concatenate([row["Muon.Phi"][row["Muon.PT"] > 25.0], row["Electron.Phi"][row["Electron.PT"] > 27]])[0])
        self._add_column("px_lepton", lambda row: row["lepton_pt"] * np.cos(row["lepton_phi"]))
        self._add_column("py_lepton", lambda row: row["lepton_pt"] * np.sin(row["lepton_phi"]))
        self._add_column("pz_lepton", lambda row: row["lepton_pt"] * np.sinh(row["lepton_eta"]))
        self._add_column("E_lepton", lambda row: row["lepton_pt"] * np.cosh(row["lepton_eta"]))

        # flatten the properties of the MET
        self._add_column("pxMET", lambda row: row["MET"] * np.cos(row["MissingET.Phi"][0]))
        self._add_column("pyMET", lambda row: row["MET"] * np.sin(row["MissingET.Phi"][0]))

        def W_constraint_1(row):
            mW = 80.37
            X = mW**2 + 2 * row["px_lepton"] * row["pxMET"] + 2 * row["py_lepton"] * row["pyMET"]
            if abs(X) > abs(2 * row["lepton_pt"] * row["MET"]):
                pzMET = 1 / (2 * row["lepton_pt"] ** 2) * (X * row["pz_lepton"] + np.sqrt(X**2 - 4 * (row["lepton_pt"] * row["MET"])**2))
            else:
                # get an imaginary soluation, instead rescale the MET to make the discriminant zero
                alpha = mW**2 / (2 * row["lepton_pt"] * row["MET"] - 2 * row["px_lepton"] * row["pxMET"] - 2 * row["py_lepton"] * row["pyMET"])
                pzMET = 1 / (2 * row["lepton_pt"] ** 2) * row["pz_lepton"] * (mW**2 + 2 * row["px_lepton"] * row["pxMET"] * alpha + 2 * row["py_lepton"] * row["pyMET"] * alpha)
            return pzMET

        def W_constraint_2(row):
            mW = 80.37
            X = mW**2 + 2 * row["px_lepton"] * row["pxMET"] + 2 * row["py_lepton"] * row["pyMET"]
            if abs(X) > abs(2 * row["lepton_pt"] * row["MET"]):
                pzMET = 1 / (2 * row["lepton_pt"] ** 2) * (X * row["pz_lepton"] - np.sqrt(X**2 - 4 * (row["lepton_pt"] * row["MET"])**2))
            else:
                # get an imaginary soluation, instead rescale the MET to make the discriminant zero
                alpha = mW**2 / (2 * row["lepton_pt"] * row["MET"] - 2 * row["px_lepton"] * row["pxMET"] - 2 * row["py_lepton"] * row["pyMET"])
                pzMET = 1 / (2 * row["lepton_pt"] ** 2) * row["pz_lepton"] * (mW**2 + 2 * row["px_lepton"] * row["pxMET"] * alpha + 2 * row["py_lepton"] * row["pyMET"] * alpha)
            return pzMET

        def delta_phi(phi1, phi2):
            dphi = abs(phi1 - phi2)
            if dphi > np.pi:
                dphi = 2 * np.pi - dphi

            return dphi

        self._add_column("pzMET1", W_constraint_1)
        self._add_column("pzMET2", W_constraint_2)
        self._add_column("EMET1", lambda row: np.sqrt(row["MET"]**2 + row["pzMET1"]**2))
        self._add_column("EMET2", lambda row: np.sqrt(row["MET"]**2 + row["pzMET2"]**2))

        # flatten the properties of the two b-jets
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
        self._add_column("pTBB", lambda row: np.sqrt(row["pxBB"]**2 + row["pyBB"]**2))
        self._add_column("EBB", lambda row: row["EB1"] + row["EB2"])
        self._add_column("phiBB", lambda row: np.arctan2(row["pyBB"], row["pxBB"]))        
        self._add_column("etaBB", lambda row: np.arcsinh(row["pzBB"] / row["pTBB"]))

        self._add_column("dEtaBB", lambda row: abs(row["etaB1"] - row["etaB2"]))
        self._add_column("dRBB", lambda row: np.sqrt((row["etaB1"] - row["etaB2"]) ** 2 + delta_phi(row["phiB1"], row["phiB2"]) ** 2))
        
        # compute the invariant mass of the two b-jets
        self._add_column("mBB", lambda row: np.sqrt(row["EBB"] ** 2 - row["pxBB"] ** 2 - row["pyBB"] ** 2 - row["pzBB"] ** 2))

        # attempt to reconstruct the top from the neutrino, lepton and one of the b-jets
        self._add_column("mtop1", lambda row: np.sqrt((row["E_lepton"] + row["EMET1"] + row["EB1"]) ** 2 - 
                                                      (row["px_lepton"] + row["pxMET"] + row["pxB1"]) ** 2 - 
                                                      (row["py_lepton"] + row["pyMET"] + row["pyB1"]) ** 2 - 
                                                      (row["pz_lepton"] + row["pzMET1"] + row["pzB1"]) ** 2))
        
        self._add_column("mtop2", lambda row: np.sqrt((row["E_lepton"] + row["EMET2"] + row["EB2"]) ** 2 - 
                                                      (row["px_lepton"] + row["pxMET"] + row["pxB2"]) ** 2 - 
                                                      (row["py_lepton"] + row["pyMET"] + row["pyB2"]) ** 2 - 
                                                      (row["pz_lepton"] + row["pzMET2"] + row["pzB2"]) ** 2))
        
        self._add_column("mtop", lambda row: np.min([row["mtop1"], row["mtop2"]]))
        self._add_column("pzMET", lambda row: np.array([row["pzMET1"], row["pzMET2"]])[np.argmin([row["mtop1"], row["mtop2"]])]) # use the solution for METz that minimizes mtop

        # put the final signal region cut on mbb and mtop
        self._select(lambda row: row["mBB"] >= 75 or row["mtop"] <= 225)

        # compute transverse momentum of the V boson (as sum of MET and the lepton)
        self._add_column("pTV", lambda row: np.sqrt((row["pxMET"] + row["px_lepton"])**2 + (row["pyMET"] + row["py_lepton"])**2))
        self._add_column("phiV", lambda row: np.arctan2(row["pyMET"] + row["py_lepton"], row["pxMET"] + row["px_lepton"]))
        self._add_column("etaV", lambda row: np.arcsinh((row["pzMET"] + row["lepton_pt"]) / row["pTV"]))
        self._add_column("dEtaVH", lambda row: abs(row["etaV"] - row["etaBB"]))

        self._add_column("dPhiVH", lambda row: delta_phi(row["phiV"], row["phiBB"]))
        self._add_column("dPhilb", lambda row: np.min([delta_phi(row["lepton_phi"], row["phiB1"]), delta_phi(row["lepton_phi"], row["phiB2"])]))
        self._add_column("mTW", lambda row: np.sqrt(2 * row["lepton_pt"] * row["MET"] * (1 - np.cos(delta_phi(row["lepton_phi"], row["MissingET.Phi"][0])))))

        if self.df is not None:
            return self.df[self.output_branches]
        else:
            return None
