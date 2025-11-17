from rdkit import Chem

smiles = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"  # ejemplo: etanol
mol = Chem.MolFromSmiles(smiles)

# Convertir a InChI
inchi = Chem.MolToInchi(mol)
print("InChI:", inchi)

# Convertir a InChIKey
inchikey = Chem.InchiToInchiKey(inchi)
print("InChIKey:", inchikey)
