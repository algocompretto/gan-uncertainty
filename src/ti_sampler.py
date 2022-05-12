from email import header
import numpy as np
import pygeostat as gs

numpy_array = np.loadtxt("/mnt/c/Users/gustavo.scholze/gan-for-mps/bin/gan_results.out")
header_name = [f'ti_{idx}' for idx in range(numpy_array.shape[1])]
header_name = f'\n'.join(header_name)

# Save again with header
np.savetxt(fname="/mnt/c/Users/gustavo.scholze/gan-for-mps/bin/gan_results.out",
            X=numpy_array,
            header=f"gan_results\n{numpy_array.shape[1]}\n{header_name}", comments="")

conditional_samples = "/mnt/c/Users/gustavo.scholze/gan-for-mps/bin/samples50.out"
generated_images = "/mnt/c/Users/gustavo.scholze/gan-for-mps/bin/gan_results.out"

ti_selection_p = gs.Program("/mnt/c/Users/gustavo.scholze/gan-for-mps/bin/ti-selector.exe")


parstr = """                                    Parameters
                                   ************
 
 START OF PARAMETERS:
 {conditional_samples}                                       - samples datafile name
 50                                              - sample number
 250 0.5 1                                         - nx, xmn, xsiz; in sample migration grid
 250 0.5 1                                         - ny, ymn, ysiz; in sample migration grid
 1   0.5 1                                         - nz, zmn, zsiz; in sample migration grid
 {generated_images}                                        - training images datafile name
 250 0.5 1                                         - nx, xmn, xsiz; in training images
 250 0.5 1                                         - ny, ymn, ysiz; in training images
 1   0.5 1                                         - nz, zmn, zsiz; in training images
 25 25 0                                           - search radii in x,y and z directions
 1                                                - minimum event order to analyze
 10                                                - maximum event order to analyze
 0                                                 - compatibility mode: 0 for relative, 1 for absolute
 1                                                 - column (for absolute mode)
 0.5                                                 - tolerance parameter
 0                                                 - variable type
 0.1                                               - acceptance threshold (continuous variables)
 0.45                                                 - training image fraction to check
 12345                                             - random seed
 output.out                                        - output file name
"""

pars = dict(conditional_samples=conditional_samples,
            generated_images=generated_images)

ti_selection_p.run(parstr=parstr.format(**pars),
                    nogetarg=pars)

