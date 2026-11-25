# QMaSIF Real 3D Structural Interpretability Visualization (Unseen Complex 1AK4)
reinitialize
load 1AK4_qmasif_annotated.pdb, qmasif_1ak4
bg_color white
set ray_opaque_background, on
set antialias, 2
set ray_shadows, 0

hide everything
select target_chain, chain B
select partner_chain, chain C

# Render target protein molecular surface colored by learned QMaSIF interface score
show surface, target_chain
spectrum b, blue_white_red, target_chain, minimum=0, maximum=100
set surface_quality, 1

# Render true experimental binding partner as translucent green cartoon for ground-truth comparison
show cartoon, partner_chain
color palegreen, partner_chain
set cartoon_transparency, 0.25, partner_chain

# Highlight ground-truth interface contacts (<= 5A) optionally via selection
select true_interface, target_chain within 5.0 of partner_chain

orient target_chain
zoom qmasif_1ak4, 2
# Uncomment the line below inside PyMOL to render high-resolution publication figure:
# ray 1800, 1400
# png Fig5b_PyMOL_1AK4_QMaSIF.png, dpi=300
