'''
This script is for parsing and extracting the text data from the EfCamDat xml file.
We need to manually make sure that we cover as many topics as possible for each level, making the topic distribution fairly even
'''
import xml.etree.ElementTree as ET
import random
import pickle
from collections import defaultdict

# set up the 'containers'
# SCRIPTS is a default dict structure whose default value is another defaultdict whose default value in turn is an empty list.
# desired object at the end:
# {'level1':{'unit1':[text1, text2 ...], 'unit2':[text1, text2...]...}, 'level2':{'unit1':[text1, text2 ...], 'unit2':[text1, text2...]...}, 'level3':{'unit1':[text1, text2 ...], 'unit2':[text1, text2...]...}...}
# eventually TEXTS will contain the final data along with CEFR labels

SCRIPTS = defaultdict(lambda: defaultdict(list))
TEXTS = []
# mapping from EfCamDatto CEFR levels
LEVEL_MAP = {
'1':'a1',
'2':'a1',
'3':'a1',
'4':'a2',
'5':'a2',
'6':'a2',
'7':'b1',
'8':'b1',
'9':'b1',
'10':'b2',
'11':'b2',
'12':'b2',
'13':'c1',
'14':'c1',
'15':'c1',
'16':'c2',
}

######################################################################################################
############# Fetching data from the xml file ################

tree = ET.parse('raw_data.xml')
root = tree.getroot()
# print(root.tag)

# root[0] is meta, root[1] is collection of writings
writings = root[1]
counter = 0 # counting num of scripts being saved

for writing in writings.getchildren():
    try:
    	level = writing.attrib['level']
    	unit = writing.attrib['unit']
    	# 4th child of writing should be the text, but check to be sure
    	text = writing[4]
    	if text.tag == 'text':
    		# populate SCRIPTS with text data, matched with correct level and unit info
    		# SCRIPTS[key][key] is a list, empty by default
    		# sometimes text is interupted by breaks, <br />, hence use itertext() to join them
    		SCRIPTS[level][unit].append(' '.join(text.itertext()))
    		counter += 1
    	else:
    		raise ValueError('Problem with writing', writing.attrib['id'])
    except Exception as e:
        print(type(e))
        print(e)

print('len(SCRIPTS:), this is number of levels', len(SCRIPTS))
for k, v in SCRIPTS.items():
	print('level ' + k + '\t' + str(len(v)) + ' units')
	for unit, text in v.items():
		print('unit ' + unit + '\t' + str(len(text)) + ' scripts')
print('{} in total'.format(counter))
print('Parsing raw file complete\n')

############### Getting a random selection and saving in TEXTS ###########################

# keep track of how many samples from each unit we end up using
sample_counts = []
for level, content in SCRIPTS.items():
    #print(This is about level', level)
    for unit, samples in content.items():
        # samples is list of scripts
        # from each unit of each level, take a random 200 sample or, if there are fewer than 100, all of them
        idices = random.sample(range(len(samples)), min(250, len(samples)))
        sample_counts.append((level, unit, str(len(idices))))
        for idx in idices:
            # level is mapped from EfCamDat level to CEFR
            TEXTS.append((samples[idx], LEVEL_MAP[level]))

print('Collected {} samples in total'.format(len(TEXTS)))
# print(TEXTS[-10:])

############### Pickling data and getting record of sample counts per unit ################

fop = open('efcamdat-data.pickle', 'wb')
pickle.dump(TEXTS, fop)
fop.close()

sample_counts = sorted(sample_counts, key=lambda x:(int(x[0]), int(x[1])))
with open('sample_counts.txt', 'w') as fo:
    for item in sample_counts:
        fo.write(item[0] + ':' + item[1] + '\t' + item[2] + '\n')










###########################################
