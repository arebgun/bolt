#!/usr/bin/env python
from __future__ import division

# from random import random
import sys
sys.path.insert(1,"..")
from myrandom import random
choice = random.choice
random = random.random

from semantics import run
import matplotlib.pyplot as plt
import utils
from planar import Vec2
from itertools import product
from utils import (logger, m2s, count_lmk_phrases, zrm_entropy,
	shannon_entropy_of_probs, min_entropy)
import numpy as np
import shelve
from location_from_sentence import get_all_sentence_posteriors, get_tree_probs2
from nltk.tree import ParentedTree
from parse import get_modparse

from models import CProduction, CWord
from features import featureList
from scikits.learn import svm

import pprint
pp = pprint.PrettyPrinter(indent=4)

step = 0.02

def precompute(scene_desc):
	scene, speaker, image = scene_desc

	utils.scene.set_scene(scene,speaker)
	scene_bb = scene.get_bounding_box()
	scene_bb = scene_bb.inflate( Vec2(scene_bb.width*0.5,scene_bb.height*0.5) )
	# table = scene.landmarks['table'].representation.get_geometry()
	loi = [lmk for lmk in scene.landmarks.values() if lmk.name != 'table']
	all_heatmaps_tupless, xs, ys = speaker.generate_all_heatmaps(scene, step=step, loi=loi)

	loi_infos = []
	all_meanings = set()
	for obj_lmk,all_heatmaps_tuples in zip(loi, all_heatmaps_tupless):

		lmks, rels, heatmapss = zip(*all_heatmaps_tuples)
		meanings = zip(lmks,rels)
		# print meanings
		all_meanings.update(meanings)
		loi_infos.append( (obj_lmk, meanings, heatmapss) )

	all_heatmaps_tuples = speaker.generate_all_heatmaps(scene, step=step)[0][0]
	# landmarks = list(set(zip(*all_heatmaps_tuples)[0]))


	# Calculate each meanings applicability to each obj_lmk in each scene
	logger( "Calculating meaning applicabilities" )
	object_meaning_applicabilities = {}
	for obj_lmk, ms, heatmapss in loi_infos:
		indexes = obj_lmk.representation.contains_points(list(product(xs,ys)))
		for m,(h1,h2) in zip(ms, heatmapss):
			# ps = [p for (x,y),p in zip(list(product(xs,ys)),h1) if obj_lmk.representation.contains_point( Vec2(x,y) )]
			ps = h1[indexes]
			if m not in object_meaning_applicabilities:
				object_meaning_applicabilities[m] = {}
			object_meaning_applicabilities[m][obj_lmk] = sum(ps)/len(ps)

	logger( "Normalizing across objects" )
	# k = len(loi)
	for meaning_dict in object_meaning_applicabilities.values():
		total = sum( meaning_dict.values() )
		if total != 0:
			for obj_lmk in meaning_dict.keys():
				meaning_dict[obj_lmk] *= meaning_dict[obj_lmk]/total

	sorted_meaning_lists = {}
	some_other_dict = {}

	logger( "Sorting meaning applicabilities" )
	for m in object_meaning_applicabilities.keys():
		for obj_lmk in object_meaning_applicabilities[m].keys():
			if obj_lmk not in sorted_meaning_lists:
				sorted_meaning_lists[obj_lmk] = []
				some_other_dict[obj_lmk] = []
			sorted_meaning_lists[obj_lmk].append( (object_meaning_applicabilities[m][obj_lmk], m) )
			# some_other_dict[obj_lmk].append( (0, m) )
	for obj_lmk in sorted_meaning_lists.keys():
		sorted_meaning_lists[obj_lmk].sort(reverse=True)
	logger( "Done with sorted_meaning_lists" )

	return sorted_meaning_lists, all_meanings

f = shelve.open('working.shelf')
# scene_directory = 'static_scenes'
# scene_descs = run.read_scenes(scene_directory,normalize=True,image=True)
# listss = map(precompute,scene_descs)
# [scene_desc.extend([sml,am]) for scene_desc,(sml,am) in zip(scene_descs,listss)]

# f['scene_descs'] = scene_descs
scene_descs = f['scene_descs']

cw_db = CWord.get_word_counts()
count = cw_db.count()
print count
items = []
poss = set()
rel_counts = {}
for cword in cw_db.all():
	poss.add(cword.pos)
	rel = (cword.relation,cword.relation_distance_class,cword.relation_degree_class)
	if rel in rel_counts:
		rel_counts[rel] += cword.count
	else:
		rel_counts[rel] = cword.count
ckeys, ccounts = zip(*rel_counts.items())
ccounts = np.array(ccounts, dtype=float)
ccount_probs = ccounts/ccounts.sum()
rel_priors = dict(zip(ckeys,ccount_probs))

for pos in poss:
	ccounter = {}
	cw_db = CWord.get_word_counts(pos=pos)
	for cword in cw_db.all():
		production = (cword.pos,cword.word)
		if production in ccounter: ccounter[production] += cword.count
		else: ccounter[production] = cword.count + 1
	ckeys, ccounts = zip(*ccounter.items())
	ccounts = np.array(ccounts, dtype=float)
	ccount_probs = ccounts/ccounts.sum()
	items.extend(zip(ckeys,ccount_probs))
priors = dict(items)
columns = ['landmark_class', 
		   'landmark_orientation_relations',
		   'landmark_color',
		   'relation',
		   'relation_distance_class',
		   'relation_degree_class']
column_values = {'landmark_class':set(), 
				 'landmark_orientation_relations':set(),
				 'landmark_color':set(),
				 'relation':set(),
				 'relation_distance_class':set(),
				 'relation_degree_class':set()}
cw_db = CWord.get_word_counts()
for column in columns:
	ents = []
	for column in columns:
		ccounter = {}
		for cword in cw_db.all():
			column_values[column].add(getattr(cword,column))
			if getattr(cword,column) in ccounter: ccounter[getattr(cword,column)] += cword.count
			else: ccounter[getattr(cword,column)] = cword.count + 1
		ckeys, ccounts = zip(*ccounter.items())
		ccounts = np.array(ccounts, dtype=float)
		ccount_probs = ccounts/ccounts.sum()
		priors = dict(priors.items()+zip(ckeys,ccount_probs))
		w_entropy = -np.sum( (ccount_probs * np.log(ccount_probs)) )
		ents.append(w_entropy)
all_ents = dict(zip(columns,ents))
pp.pprint(all_ents)
# f['columns'] = columns
# f['column_values'] = column_values
# f['all_ents'] = all_ents
# f['priors'] = priors
# f['rel_priors'] = rel_priors

# columns  = f['columns']
# column_values = f['column_values']
# all_ents = f['all_ents']
# priors   = f['priors']
# rel_priors = f['rel_priors']

column_fields = ['lmk_class',
				 'lmk_ori_rels',
				 'lmk_color',
				 'rel',
				 'rel_dist_class',
				 'rel_deg_class']
column_fields = dict(zip(columns,column_fields))
f.close()
# exit()

clf = svm.SVC(kernel='linear')
X = []
y = []
weights = []
answers = []
epsilon = 0.1
# for scene,speaker,image,sorted_meaning_lists,all_meanings in [scene_descs[0],scene_descs[4],scene_descs[1],scene_descs[2],scene_descs[3]]:#scene_descs:
(scene,speaker,image,sorted_meaning_lists,all_meanings) = scene_descs[0]
utils.scene.set_scene(scene,speaker)
plt.ion()
plt.imshow(image)
plt.show()
while True:
	obj_lmks = [lmk for lmk in scene.landmarks.values() if lmk.name != 'table']

	sentence = raw_input('Sentence: ')
		
	posteriors = get_all_sentence_posteriors(sentence, all_meanings, printing=False)
	posts = [(posteriors[lmk]*posteriors[rel],posteriors[lmk],posteriors[rel],m2s(lmk,rel)) for lmk,rel in all_meanings]
	for post, postl, postr, m in sorted(posts)[-10:]:
		print post,postl,postr,m

	print
	posts, postls, postrs, ms = zip(*posts)
	posts = np.array(posts)
	posts /= posts.sum()
	posts = zip(posts,postls,postrs,ms)
	for post, postl, postr, m in sorted(posts)[-10:]:
		print post,postl,postr,m

	print
	logger( 'parsing ...' )
	_, modparse = get_modparse(sentence)
	logger( modparse )
	t = ParentedTree.parse(modparse)
	print '\n%s\n' % t.pprint()
	num_ancestors = count_lmk_phrases(t) - 1
	lmks, rels = zip(*all_meanings)
	lmks = list(set(lmks))
	rels = list(set(rels))

	posts = np.array([posteriors[rel] for rel in rels])
	posts /= posts.sum()
	# best_lmk = sorted(zip(posts,lmks))[-1]
	sorted_rels = sorted(zip(posts,rels), reverse=True)
	for p,relation in sorted_rels:
		print p,m2s(lmks[0],relation)
	print
	print
	posts = np.array([posteriors[lmk] for lmk in lmks])
	posts /= posts.sum()
	# best_lmk = sorted(zip(posts,lmks))[-1]
	sorted_lmks = sorted(zip(posts,lmks), reverse=True)
	for p,landmark in sorted_lmks:
		print p,m2s(landmark,relation)
	print

	for i,lmk in enumerate(obj_lmks):
		print i+1,lmk
	trajector = obj_lmks[int(raw_input('Trajector? '))-1]
	real_landmark = obj_lmks[int(raw_input('Landmark? '))-1]

	# lmk_probs = []
	# for i,lmk in enumerate(lmks):
	# 	if lmk.get_ancestor_count() != num_ancestors:
	# 		p = 0
	# 	else:
	# 		pc,ec,lrpc,_ = get_tree_probs2(t[1], lmk, golden=False, printing=False)
	# 		# print len(pc),len(ec),len(lrpc)
	# 		# for p,e,lrp in zip(pc,ec,lrpc):
	# 		# 	print p,e,lrp
	# 		p = np.prod(pc)
	# 	lmk_probs.append(p)
	# sorted_lmks = sorted(zip(lmk_probs,lmks),reverse=True)

	for p,landmark in sorted_lmks:

		if p < epsilon:
			break

		# landmark = sorted_lmks[-1][1]
		perspective = speaker.get_head_on_viewpoint(landmark)

		features = {}
		for obj in obj_lmks:
			features[obj] = [feature.measure(perspective,landmark,obj)
							 for feature in featureList]

		pp.pprint(features)

		new_X,new_y,new_weights,new_answers = zip(*[(features[obj],obj==trajector,p,landmark==real_landmark and obj==trajector) for obj in obj_lmks])
		X.extend(new_X)
		y.extend(new_y)
		weights.extend(new_weights)
		answers.extend(new_answers)
	print X
	print y
	print weights

	if hasattr(clf,'shape_fit_'):
		print clf.predict(X)
	print clf.fit(X, y, sample_weight=weights)
	predictions = np.array(clf.predict(X),dtype=bool)
	# print predictions
	correct = predictions == y
	# print correct
	# print sum(correct)
	# print np.dot(correct,weights)
	print predictions
	print answers

	print 'precision: ', sum((predictions==answers)[predictions])/sum(predictions)
	print 'recall: ', sum((predictions==answers)[predictions])/sum(answers)
	print 
	print clf.support_vectors_


	prev_word = None
	for pos in t.treepositions('leaves'):
		rhs = t[pos]
		lhs = t[pos[:-1]].node
		parent = t[pos[:-2]].node
		print parent, lhs, rhs

		cw_db = CWord.get_word_counts(pos=lhs,
									  word=rhs,
									  prev_word=prev_word)
		prev_word = rhs

		count = cw_db.count()
		if count > 0:
			N=0
			ccounter = {}
			rel_fieldss = set()
			for cword in cw_db.all():
				N+=cword.count
				rel_fields = (cword.relation,cword.relation_distance_class,cword.relation_degree_class)

				if rel_fields in ccounter:
					ccounter[rel_fields] += cword.count
				else:
					ccounter[rel_fields] = cword.count
			ckeys, ccounts = zip(*ccounter.items())
			ccounts = np.array(ccounts, dtype=float)
			ccount_probs = ccounts/ccounts.sum()
			probs = dict(zip(ckeys,ccount_probs))

			cvalues = rel_priors.keys()
			ccount_probs, worst_probs = zip( *[((probs[key] if key in probs else 0.0),rel_priors[key]) for key in cvalues] )

			shannon = shannon_entropy_of_probs(ccount_probs)
			normed_shannon = shannon_entropy_of_probs(ccount_probs,worst_probs=worst_probs)
			laplace_shannon = shannon_entropy_of_probs(ccount_probs,N=N)
			normed_laplace_shannon = shannon_entropy_of_probs(ccount_probs,N=N,worst_probs=worst_probs)

			zrm = zrm_entropy(ccount_probs, worst_probs, N)
			non_laplace_zrm = zrm_entropy(ccount_probs, worst_probs)

			min_ent = min_entropy(ccount_probs)
			normed_min_ent = min_entropy(ccount_probs,worst_probs=worst_probs)
			laplace_min_ent = min_entropy(ccount_probs,N=N)
			normed_laplace_min_ent = min_entropy(ccount_probs,N=N,worst_probs=worst_probs)

			normed = ccount_probs*np.array(worst_probs)
			normed = normed/normed.sum()

			print 'Shannon entropy:', shannon
			print 'Normed Shannon entropy:',normed_shannon
			print 'Laplace Shannon entropy:',laplace_shannon
			print 'Normed Laplace Shannon entropy:',normed_laplace_shannon
			print 'ZRM entropy:', zrm
			print 'Non-Laplace ZRM:',non_laplace_zrm
			print 'Min-entropy:', min_ent
			print 'Normed Min-entropy:',normed_min_ent
			print 'Laplace Min-entropy:',laplace_min_ent
			print 'Normed Laplace Min-entropy:',normed_laplace_min_ent
			print 'Direct p(rel|(%s,%s))' % (lhs,rhs)
			print dict(zip(cvalues,ccount_probs))
			print 'Prior p(rel)'
			print dict(zip(cvalues,worst_probs))
			print 'Normed p(rel|(%s,%s))' % (lhs,rhs)
			print dict(zip(cvalues,normed))
			print N
		print
		print
		print

		# count = cw_db.count()
		# print count
		# ents = []
		# norm_ents = []
		# for column in columns:
		# 	if count > 0:
		# 		N=0
		# 		ccounter = {}
		# 		for cword in cw_db.all():
		# 			N += cword.count
		# 			if getattr(cword,column) in ccounter: ccounter[getattr(cword,column)] += cword.count
		# 			else: ccounter[getattr(cword,column)] = cword.count + 1
		# 		ckeys, ccounts = zip(*ccounter.items())
		# 		ccounts = np.array(ccounts, dtype=float)
		# 		ccount_probs = ccounts/ccounts.sum()
		# 		probs = dict(zip(ckeys,ccount_probs))

		# 		# Got normal p(feat|(lhs,rhs)), now calculate it Bayes' way
		# 		# bayes_probs = []
		# 		# ccounter2 = {}
		# 		# for ckey in ckeys:
		# 		# 	# Calculate p(())
		# 		# 	cw_db2 = CWord.get_word_counts(pos=lhs,**dict([(column_fields[column],
		# 		# 											ckey)]))
		# 		# 	for cword in cw_db2.all():
		# 		# 		production = (cword.pos,cword.word)
		# 		# 		if production in ccounter2: ccounter2[production] += cword.count
		# 		# 		else: ccounter2[production] = cword.count + 1
		# 		# 	ckeys2, ccounts2 = zip(*ccounter2.items())
		# 		# 	print ckeys2
		# 		# 	ccounts2 = np.array(ccounts2, dtype=float)
		# 		# 	probs = ccounts2/ccounts2.sum()
		# 		# 	pwordgivenfeat = dict(zip(ckeys2,probs))[(lhs,rhs)]
		# 		# 	print
		# 		# 	print
		# 		# 	print ckey
		# 		# 	print '  ',pwordgivenfeat
		# 		# 	print '  ',priors[ckey]
		# 		# 	print '  ',priors[(lhs,rhs)]
		# 		# 	bayes_probs.append( pwordgivenfeat*
		# 		# 						priors[ckey]/
		# 		# 						priors[(lhs,rhs)])

		# 		# w_entropy = -np.sum( (ccount_probs * np.log(ccount_probs)) )
		# 		cvalues = list(column_values[column])
		# 		ccount_probs, worst_probs = zip( *[((probs[value] if value in probs else 0.0),priors[value]) for value in cvalues] )
		# 		# worst_probs = [priors[ckey] for ckey in ckeys]
		# 		w_entropy = zrm_entropy(N, ccount_probs, worst_probs)
		# 		print
		# 		print w_entropy
		# 		print 'Direct p(%s|(%s,%s))' % (column,lhs,rhs)
		# 		print dict(zip(cvalues, ccount_probs))
		# 		print 'Prior p(%s)' % column
		# 		print dict(zip(cvalues,worst_probs))
		# 		print N
		# 		# print 'Bayes p(%s|(%s,%s))' % (column,lhs,rhs)
		# 		# print dict(zip(ckeys,bayes_probs))

		# 		ents.append(w_entropy)
		# 		norm_ents.append(w_entropy/all_ents[column])
		# 	else:
		# 		ents.append(float('inf'))
		# pp.pprint( dict(zip(columns,ents)) )
		# pp.pprint( dict(zip(columns,norm_ents)) )




								  # lmk_class=lmk_class,
								  # lmk_ori_rels=lmk_ori_rels,
								  # lmk_color=lmk_color,
								  # rel=rel_class,
								  # rel_dist_class=dist_class,
								  # rel_deg_class=deg_class,
								  # golden=golden)

	# sentence = raw_input('Sentence: ')
