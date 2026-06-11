import shutil
shutil.copyfile('/Users/ncc1031a/Documents/VSCode/PythonProgramDesign/week14/sample.txt','/Users/ncc1031a/Documents/VSCode/PythonProgramDesign/week14/sample1.txt')
shutil.make_archive('hamood', 'zip', '/Users/ncc1031a/Documents/VSCode/PythonProgramDesign/week14')   
shutil.unpack_archive('hamood.zip', '/Users/ncc1031a/Documents/VSCode/PythonProgramDesign/week14')