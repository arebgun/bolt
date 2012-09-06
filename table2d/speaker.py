from numpy import (
    array,
    arange,
    log,
    argmin,
    set_printoptions,
    # random,
    copy as npcopy,
    exp
)
import sys
sys.path.append("..")
from myrandom import random
choice = random.choice
from myrandom import nprandom as random
# from random import choice

from matplotlib import pyplot as plt
from itertools import product
from planar import Vec2

import language_generator

from landmark import (
    PointRepresentation,
    LineRepresentation,
    GroupLineRepresentation,
    RectangleRepresentation,
    Landmark,
)

from relation import (
    DistanceRelationSet,
    ContainmentRelationSet,
    OrientationRelationSet,
    Measurement,
    Degree
)


class Speaker(object):
    def __init__(self, location):
        self.location = location

    def get_head_on_viewpoint(self, landmark):
        axes = landmark.get_primary_axes()
        if len(axes) > 0:
            axis = axes[ argmin([axis.distance_to(self.location) for axis in axes]) ]
            return axis.project(self.location)
        else:
            print "Not getting head on viewpoint!!!"
            return self.location

    def sample_meaning(self, trajector, scene, max_level=-1):

        scenes = scene.get_child_scenes(trajector) + [scene]

        all_landmarks = []

        for s in scenes:
            for scene_lmk in s.landmarks.values():

                # Don't want to use the trajector as landmark
                if scene_lmk == trajector:
                    continue

                all_landmarks.append([s, scene_lmk])

                representations = [scene_lmk.representation]
                representations.extend(scene_lmk.representation.get_alt_representations())

                for representation in representations:
                    for lmk in representation.get_landmarks(max_level):
                        all_landmarks.append([s, lmk])

        sceness, landmarks = zip( *all_landmarks )

        sampled_landmark, sl_prob, sl_ent, head_on = self.sample_landmark( landmarks, trajector )
        # print '   ', sampled_landmark, sl_prob, sl_ent

        sampled_relation, sr_prob, sr_ent = self.sample_relation( trajector, scene.get_bounding_box(), head_on, sampled_landmark, step=0.1 )
        # print '   ', sampled_relation, sr_prob, sr_ent
        sampled_relation = sampled_relation( head_on, sampled_landmark, trajector )

        return sampled_landmark, sampled_relation, head_on

    def describe(self, trajector, scene, visualize=False, max_level=-1, delimit_chunks=False):

        sampled_landmark, sampled_relation, head_on = self.sample_meaning(trajector, scene, max_level)

        description = str(trajector.representation.middle) + '; ' + language_generator.describe(head_on, trajector, sampled_landmark, sampled_relation, delimit_chunks)
        print description

        if visualize: self.visualize(scene, trajector, head_on, sampled_landmark, sampled_relation, description, 0.01)
        return description

    def get_all_meaning_descriptions(self, trajector, scene, sampled_landmark=None, sampled_relation=None, head_on=None, max_level=-1):
        if sampled_landmark is None or sampled_relation is None or head_on is None:
            sampled_landmark, sampled_relation, head_on = self.sample_meaning(trajector, scene, max_level)
        return language_generator.get_all_descriptions(head_on, trajector, sampled_landmark, sampled_relation)

    def communicate(self, scene, visualize=False, max_level=-1, delimit_chunks=False):
        all_landmarks = []
        all_relations = []

        for scene_lmk in scene.landmarks.values():
            all_landmarks.append(scene_lmk)

            representations = [scene_lmk.representation]
            representations.extend(scene_lmk.representation.get_alt_representations())

            for representation in representations:
                all_landmarks.extend(representation.get_landmarks(max_level))

        for rset in [DistanceRelationSet,ContainmentRelationSet, OrientationRelationSet]:
            all_relations.extend(rset.relations)

        sampled_landmark = choice(all_landmarks)
        sampled_relation = choice(all_relations)
        perspective = self.get_head_on_viewpoint(sampled_landmark)
        self.set_orientations(sampled_landmark, perspective)

        trajector = self.sample_point_trajector( scene.landmarks['table'].representation.get_geometry().bounding_box,
                                                 sampled_relation,
                                                 perspective,
                                                 sampled_landmark)

        print sampled_landmark, self.get_landmark_probability( sampled_landmark, all_landmarks, trajector )
        print sampled_relation, self.get_relation_probability( sampled_relation, trajector, scene.get_bounding_box(), perspective, sampled_landmark, step=0.1)

        sampled_relation = sampled_relation(perspective, sampled_landmark, trajector)
        description = str(trajector) + '; ' + language_generator.describe(perspective, trajector, sampled_landmark, sampled_relation, delimit_chunks)
        print description

        if visualize: self.visualize(scene, trajector, perspective, sampled_landmark, sampled_relation, description, 0.1)

    def set_orientations(self, landmark, perspective):
        options = set()
        if landmark.parent and landmark.parent.parent_landmark:
            middle_lmk = Landmark('', PointRepresentation(landmark.parent.middle), landmark.parent, None)
            options = OrientationRelationSet.get_applicable_relations(perspective,
                                                                      middle_lmk,
                                                                      Landmark( None,
                                                                                PointRepresentation(landmark.representation.middle),
                                                                                None, None),
                                                                      use_distance=False)

            par_lmk = landmark.parent.parent_landmark
            if par_lmk.parent and par_lmk.parent.parent_landmark:
                par_middle_lmk = Landmark('', PointRepresentation(par_lmk.parent.middle), par_lmk.parent, None)
                trajector = Landmark('', PointRepresentation(par_lmk.representation.middle), None, None)
                par_options = OrientationRelationSet.get_applicable_relations(perspective, par_middle_lmk, trajector, use_distance=False)
            else:
                par_options = []

            options = sorted(set(options).difference(set(par_options)))
            self.set_orientations(par_lmk, perspective)

        landmark.ori_relations = map(type, options)

    def talk_to_baby(self, scene, perspectives, how_many_each=10000):

        max_recurse_level = 4
        for recurse_level in range(max_recurse_level):
            for i in range(how_many_each):
                perspective = choice(perspectives)
                self.location = perspective
                lmk = choice(scene.landmarks.values())
                level = 0
                while level < recurse_level:
                    representations = [lmk.representation]+lmk.representation.get_alt_representations()
                    landmarks = []
                    for representation in representations:
                        landmarks.extend( representation.landmarks.values() )
                    if len(landmarks) == 0:
                        break
                    lmk = choice(landmarks)
                    level += 1
                head_on = self.get_head_on_viewpoint(lmk)
                print perspective, lmk.uuid, lmk.get_description(head_on)

    def demo(self, poi, scene):

        '''
        sampled_landmark = scene.landmarks['table'].representation.landmarks['ll_corner']
        head_on = self.get_head_on_viewpoint(sampled_landmark)
        relset = DistanceRelationSet
        sampled_relation = relset.relations[0](head_on,sampled_landmark,poi)
        print 'distance',sampled_landmark.distance_to(poi)
        print 'probability', sampled_relation.is_applicable()
        description = str(poi) + '; ' + language_generator.describe(head_on, sampled_landmark, sampled_relation)
        print description
        self.visualize(scene, poi, head_on, sampled_landmark, sampled_relation, description, step=0.1)

        sampled_landmark = scene.landmarks['obj2'].representation.landmarks['r_edge']
        head_on = self.get_head_on_viewpoint(sampled_landmark)
        relset = OrientationRelationSet
        sampled_relation = relset.relations[0](head_on,sampled_landmark,poi)
        print 'distance',sampled_landmark.distance_to(poi)
        print 'probability', sampled_relation.is_applicable()
        description = str(poi) + '; ' + language_generator.describe(head_on, sampled_landmark, sampled_relation)
        print description
        self.visualize(scene, poi, head_on, sampled_landmark, sampled_relation, description, step=0.1)
        '''

    '''
    # broken!
    def get_all_descriptions(self, poi, scene, max_level=-1):

        all_desc = []
        scenes = scene.get_child_scenes(poi) + [scene]
        counter = 0
        for s in scenes:
            for scene_lmk in s.landmarks.values():
                representations = [scene_lmk.representation]
                representations.extend(scene_lmk.representation.get_alt_representations())

                for representation in representations:
                    for lmk in representation.get_landmarks(max_level)+[scene_lmk]: # we have a leaf landmark at current level
                        head_on = self.get_head_on_viewpoint(lmk)
                        lmk_desc = language_generator.get_landmark_description(head_on, lmk)

                        for relset in [DistanceRelationSet,ContainmentRelationSet, OrientationRelationSet]:

                            for relation in relset.relations: # we have a relation
                                entropy = self.get_entropy(self.get_probabilities(s, relation, head_on, lmk, 0.1)[0])
                                relation = relation(head_on, lmk, poi)
                                applies = relation.is_applicable()

                                if applies:
                                    def create_desc(adverb, prob):
                                        desc = [adverb + rd + ' ' + lmk_desc for rd in language_generator(type(relation))]
                                        score = prob*applies / entropy
                                        all_desc.append( [score, prob, entropy, s, scene_lmk, representation, lmk, relation, desc] )
                                        sys.stderr.write('[%d] %f, %f, %f\n' % (counter, score, prob, entropy))
                                        sys.stderr.flush()

                                    if hasattr(relation, 'measurement') and not isinstance(relation, VeryCloseDistanceRelation):
                                        m_probs = relation.measurement.evaluate_all(relation.distance)

                                        for prob,adverb in m_probs:
                                            create_desc(adverb + ' ', prob)
                                            counter += 1
                                    else:
                                        create_desc('', 1.0)
                                        counter += 1


        return reversed(sorted(all_desc))
    '''

    def get_probabilities_box(self, bounding_box, relation, perspective, landmark, step=0.02):
        xs = arange(bounding_box.min_point.x, bounding_box.max_point.x, step)
        ys = arange(bounding_box.min_point.y, bounding_box.max_point.y, step)
        points = array(list(product(xs,ys)))

        return self.get_probabilities_points(points, relation, perspective, landmark), points

    def get_probabilities_points(self, points, relation, perspective, landmark):
        if isinstance(relation,type):
            probabilities = relation.any_are_applicable(perspective, landmark, points)
        else:
            probabilities = relation.are_applicable(points)
        return probabilities

    def get_probabilities(self, scene, relation, perspective, landmark, step=0.02):
        scene_bb = scene.get_bounding_box()
        scene_bb = scene_bb.inflate( Vec2(scene_bb.width*0.5,scene_bb.height*0.5) )
        return self.get_probabilities_box(scene_bb, relation, perspective, landmark, step)

    def evaluate_trajector_likelihood(self, trajector, bounding_box, relation, perspective, landmark, step=0.02):
        probs, _ = self.get_probabilities_box(bounding_box, relation, perspective, landmark, step)
        rel = relation( perspective, landmark, trajector )
        trajector_prob = rel.is_applicable()
        return trajector_prob / (probs.sum() + trajector_prob) if trajector_prob else trajector_prob

    def sample_landmark(self, landmarks, trajector):
        ''' Weight by inverse of distance to landmark center and choose probabilistically  '''
        epsilon = 0.02
        distances = array([trajector.distance_to( lmk.representation )
            if not (isinstance(lmk.representation,RectangleRepresentation) and lmk.representation.contains(trajector.representation))
            else 3*epsilon for lmk in landmarks])
        # distances = array([trajector.distance_to( lmk )
        #     if not (isinstance(lmk.representation,RectangleRepresentation) and lmk.representation.contains(trajector))
        #     else min(poly_to_vec_distance(lmk.representation.get_geometry().to_polygon(), trajector.representation.location),lmk.representation.middle.distance_to(trajector.representation.location))
        #     for lmk in landmarks])
        # scores = 1.0/(distances + epsilon)**0.5
        std = .1
        scores = exp( -(distances/std)**2)
        lm_probabilities = scores/sum(scores)
        index = lm_probabilities.cumsum().searchsorted( random.sample(1) )[0]

        sampled_landmark = landmarks[index]
        head_on = self.get_head_on_viewpoint(sampled_landmark)
        self.set_orientations(sampled_landmark, head_on)

        return sampled_landmark, lm_probabilities[index], self.get_entropy(lm_probabilities), head_on

    def get_landmark_probs_for_points(self, landmarks, points, xs, ys, x, y):
        def get_probabilities(landmark, points):
            epsilon = 0.02
            distances = array([ landmark.distance_to_point(point)
                if not (isinstance(landmark.representation,RectangleRepresentation) and landmark.representation.contains_point(point))
                else 3*epsilon for point in points])
            # distances = array([ landmark.distance_to_point(point)
            #     if not (isinstance(landmark.representation,RectangleRepresentation) and landmark.representation.contains_point(point))
            #     else min(poly_to_vec_distance(landmark.representation.get_geometry().to_polygon(),point),landmark.representation.middle.distance_to(point))
            #     for point in points])
            # scores = 1.0/(distances + epsilon)**0.5
            std = .1
            scores = exp( -(distances/std)**2)
            # if scores.sum() != 0:
            print landmark, scores.sum(), max(scores)
            return scores/scores.sum()
            # else: return scores

        sum_probs = None
        prob_lists = []
        original_probs = []
        for landmark in landmarks:
            probs = get_probabilities(landmark, points)
            original_probs.append( npcopy(probs) )
            # probabilities = probs.reshape( (len(xs),len(ys)) ).T
            # plt.pcolor(x, y, probabilities, cmap = 'jet', edgecolors='none', alpha=0.7)
            # plt.colorbar()
            # plt.title(str(landmark)+" before")
            # plt.show()

            print landmark, probs.sum(), max(probs)
            if sum_probs is None: sum_probs = npcopy(probs)
            else: sum_probs += probs
            prob_lists.append( probs )

        for lmk,probs in zip(landmarks, prob_lists):

            # print
            # probabilities = probs.reshape( (len(xs),len(ys)) ).T
            # plt.pcolor(x, y, probabilities, cmap = 'jet', edgecolors='none', alpha=0.7)
            # plt.colorbar()
            # plt.title(str(lmk)+" before")
            # plt.show()

            probs /= sum_probs

            # probabilities = probs.reshape( (len(xs),len(ys)) ).T
            # plt.pcolor(x, y, probabilities, cmap = 'jet', edgecolors='none', alpha=0.7)
            # plt.colorbar()
            # plt.title(str(lmk)+" after")
            # plt.show()

        return prob_lists, original_probs

    def get_relation_probs_for_points(self, points, landmark, landmark_heatmap, original_landmark_heatmap, perspective):

        def instantiate_relations(landmark):

            bullshit_trajector = Landmark( None, PointRepresentation( Vec2(0,0) ), None )
            relations = []
            for rel in DistanceRelationSet.relations:
                for dist_class, deg_class in list(product([Measurement.NEAR,Measurement.FAR],Degree.all)):
                    relation = rel( perspective, landmark, bullshit_trajector )
                    relation.measurement.best_distance_class = dist_class
                    relation.measurement.best_degree_class = deg_class
                    relations.append(relation)

            for rel in ContainmentRelationSet.relations:
                    relation = rel( perspective, landmark, bullshit_trajector )
                    relations.append(relation)

            for rel in OrientationRelationSet.relations:
                for dist_class, deg_class in list(product([Measurement.FAR],Degree.all)) + [(Measurement.NONE,Degree.NONE)]:
                    relation = rel( perspective, landmark, bullshit_trajector )
                    relation.measurement.best_distance_class = dist_class
                    relation.measurement.best_degree_class = deg_class
                    relations.append(relation)
            return relations

        relations = instantiate_relations(landmark)
        rel_points_probs = []
        original_probs = []
        sum_probs = None
        for relation in relations:
            probs = self.get_probabilities_points(points, relation, None, None)
            if probs.sum() != 0:
                probs /= probs.sum()
            original_probs.append( npcopy(probs) )
            if sum_probs is None: sum_probs = npcopy(probs)
            else: sum_probs += probs
            rel_points_probs.append( probs )

        # normalize across relations
        for probs in rel_points_probs:
            probs /= sum_probs
            probs *= landmark_heatmap

        # TODO treat ori_relations differently

        return relations, zip(rel_points_probs, original_probs)

    def generate_all_heatmaps(self, scene, max_level=1, step=0.02, loi=[]):
        scenes = [scene]

        scene_bb = scene.get_bounding_box()
        scene_bb = scene_bb.inflate( Vec2(scene_bb.width*0.5,scene_bb.height*0.5) )

        xs = arange(scene_bb.min_point.x, scene_bb.max_point.x, step)
        ys = arange(scene_bb.min_point.y, scene_bb.max_point.y, step)
        points = array(list(product(xs,ys)))
        x = array( [list(xs-step*0.5)]*len(ys) )
        y = array( [list(ys-step*0.5)]*len(xs) ).T

        def __generate(lmk_to_exclude=None):
            all_landmarks = []

            for s in scenes:
                for scene_lmk in s.landmarks.values():
                    if scene_lmk == lmk_to_exclude: continue

                    all_landmarks.append([s, scene_lmk])

                    representations = [scene_lmk.representation]
                    representations.extend(scene_lmk.representation.get_alt_representations())

                    for representation in representations:
                        for lmk in representation.get_landmarks(max_level):
                            all_landmarks.append([s, lmk])

            sceness, landmarks = zip( *all_landmarks )

            landmark_probs, original_landmark_probs = self.get_landmark_probs_for_points(landmarks, points, xs, ys, x, y)

            lmk_rel_dict = {}
            for landmark,landmark_prob,original_landmark_prob in zip(landmarks,landmark_probs,original_landmark_probs):
                perspective = self.get_head_on_viewpoint(landmark)
                self.set_orientations(landmark, perspective)
                lmk_rel_dict[landmark] = dict( zip(*self.get_relation_probs_for_points(points, landmark, landmark_prob, original_landmark_prob, perspective)) )

            return lmk_rel_dict

        result = []
        for l in loi:
            result.append(__generate(l))
        else:
            result.append(__generate())

        return result, xs, ys


    def get_landmark_probability(self, sampled_landmark, landmarks, trajector):
        epsilon = 0.000001
        distances = array([trajector.distance_to( PointRepresentation(lmk.representation.middle) ) for lmk in landmarks])
        scores = 1.0/(array(distances)**1.5 + epsilon)
        # scores[distances == 0] = 0
        lm_probabilities = scores/sum(scores)
        return lm_probabilities[ landmarks.index(sampled_landmark) ], self.get_entropy(lm_probabilities)



    def sample_relation(self, trajector, bounding_box, perspective, landmark, step=0.02):
        """
        Sample a relation given a trajector and landmark.
        Evaluate each relation and probabilisticaly choose the one that is likely to
        generate the trajector given a landmark.
        """
        rel_scores = []
        rel_classes = []

        for s in [DistanceRelationSet, ContainmentRelationSet]:
            for rel in s.relations:
                rel_scores.append(self.evaluate_trajector_likelihood(trajector, bounding_box, rel, perspective, landmark, step))
                rel_classes.append(rel)

        ori_rel_scores = []
        for rel in OrientationRelationSet.relations:
            p = self.evaluate_trajector_likelihood(trajector, bounding_box, rel, perspective, landmark, step)
            if p > 0: ori_rel_scores.append( (p, rel) )

        if len(ori_rel_scores) > 1:
            assert( len(ori_rel_scores) == 2 )

            dists = []
            for p,rel in ori_rel_scores:
                dists.append( [rel(perspective, landmark, trajector).measurement.distance, p, rel] )
            dists = sorted(dists)
            dists[0][1] *= dists[0][0] / dists[1][0]

            rel_scores.append(dists[0][1])
            rel_scores.append(dists[1][1])
            rel_classes.append(dists[0][2])
            rel_classes.append(dists[1][2])

        rel_scores = array(rel_scores)
        rel_probabilities = rel_scores/sum(rel_scores)
        index = rel_probabilities.cumsum().searchsorted( random.sample(1) )[0]

        return rel_classes[index], rel_probabilities[index], self.get_entropy(rel_probabilities)

    def get_relation_probability(self, sampled_relation, trajector, bounding_box, perspective, landmark, step=0.02):
        rel_scores = []
        rel_classes = []

        for s in [DistanceRelationSet, OrientationRelationSet, ContainmentRelationSet]:
            for rel in s.relations:
                rel_scores.append(self.evaluate_trajector_likelihood(trajector, bounding_box, rel, perspective, landmark, step))
                rel_classes.append(rel)

        rel_scores = array(rel_scores)
        set_printoptions(threshold='nan')
        # print 'X',rel_scores
        rel_probabilities = rel_scores/sum(rel_scores)
        return rel_probabilities[ rel_classes.index(sampled_relation) ], self.get_entropy(rel_probabilities)

    def sample_point_trajector(self, bounding_box, relation, perspective, landmark, step=0.02):
        """
        Sample a point of interest given a relation and landmark.
        """
        probs, points = self.get_probabilities_box(bounding_box, relation, perspective, landmark)
        probs /= probs.sum()
        index = probs.cumsum().searchsorted( random.sample(1) )[0]
        return Landmark( 'point', Vec2( *points[index] ), None, Landmark.POINT )

    def get_entropy(self, probabilities):
        probabilities += 1e-15
        probabilities = probabilities/sum(probabilities.flatten())
        return -sum( (probabilities * log(probabilities)).flatten() )

    def visualize(self, scene, trajector, head_on, sampled_landmark, sampled_relation, description, step=0.02):

        relation = sampled_relation
        print relation
        if hasattr(relation, 'measurement'):
            print relation.measurement, relation.measurement.best_distance_class, relation.measurement.best_degree_class

        # plt.figure( figsize=(6,8) )
        #plt.subplot(1,2,1)
        scene_bb = scene.get_bounding_box()
        scene_bb = scene_bb.inflate( Vec2(scene_bb.width*0.5,scene_bb.height*0.5) )

        probabilities, points = self.get_probabilities_box( scene_bb, relation, head_on, sampled_landmark, step )
        # xs, ys = points[:,0], points[:,1]

        xs = arange(scene_bb.min_point.x, scene_bb.max_point.x, step)
        ys = arange(scene_bb.min_point.y, scene_bb.max_point.y, step)

        # probabilities = zeros(  ( len(ys),len(xs) )  )
        # for i,x in enumerate(xs):
        #     for j,y in enumerate(ys):
        #         rel = relation( head_on, sampled_landmark, Landmark('', PointRepresentation(Vec2(x,y)), None, None) )
        #         if hasattr(rel, 'measurement'):
        #             rel.measurement.best_degree_class = sampled_relation.measurement.best_degree_class
        #             rel.measurement.best_distance_class = sampled_relation.measurement.best_distance_class
        #         probabilities[j,i] = rel.is_applicable()
        #         # print rel.distance, probabilities[j,i]

        set_printoptions(threshold='nan')
        #print probabilities

        x = array( [list(xs-step*0.5)]*len(ys) )
        y = array( [list(ys-step*0.5)]*len(xs) ).T

        probabilities = probabilities.reshape( (len(xs),len(ys)) ).T

        # print probabilities

        #print self.get_entropy(probabilities)
        plt.pcolor(x, y, probabilities, cmap = 'jet', edgecolors='none', alpha=0.7)
        plt.colorbar()

        for lmk in scene.landmarks.values():
            if isinstance(lmk.representation, GroupLineRepresentation):
                xs = [lmk.representation.line.start.x, lmk.representation.line.end.x]
                ys = [lmk.representation.line.start.y, lmk.representation.line.end.y]
                plt.fill(xs,ys,facecolor='none',linewidth=2)
            elif isinstance(lmk.representation, RectangleRepresentation):
                rect = lmk.representation.rect
                xs = [rect.min_point.x,rect.min_point.x,rect.max_point.x,rect.max_point.x]
                ys = [rect.min_point.y,rect.max_point.y,rect.max_point.y,rect.min_point.y]
                plt.fill(xs,ys,facecolor='none',linewidth=2)
                plt.text(rect.min_point.x+0.01,rect.max_point.y+0.02,lmk.name)

        plt.plot(self.location.x,
                 self.location.y,
                 'bx',markeredgewidth=2)

        traj_rep = trajector.representation
        if isinstance(traj_rep, GroupLineRepresentation):
            xs = [traj_rep.line.start.x, traj_rep.line.end.x]
            ys = [traj_rep.line.start.y, traj_rep.line.end.y]
            plt.fill(xs,ys,facecolor='none',linewidth=2)
        elif isinstance(traj_rep, RectangleRepresentation):
            rect = traj_rep.rect
            xs = [rect.min_point.x,rect.min_point.x,rect.max_point.x,rect.max_point.x]
            ys = [rect.min_point.y,rect.max_point.y,rect.max_point.y,rect.min_point.y]
            plt.fill(xs,ys,facecolor='none',linewidth=2)
        elif isinstance(traj_rep, PointRepresentation):
            plt.plot(traj_rep.location.x, traj_rep.location.y, 'bo', markeredgewidth=2)

        plt.text(traj_rep.middle.x+0.01,
                 traj_rep.middle.y+0.02,'trajector')

        '''
        plt.plot(poi.x,poi.y,'rx',markeredgewidth=2)
        plt.text(poi.x+0.01,
                 poi.y+0.02,'POI')
        '''

        plt.plot(head_on.x,head_on.y,'ro',markeredgewidth=2)
        plt.text(head_on.x+0.02,head_on.y+0.01,'perspective')

        lwidth = 3
        lcolor = (0,1,0)
        if isinstance(sampled_landmark.representation, PointRepresentation):
            plt.plot(sampled_landmark.representation.location.x,
                     sampled_landmark.representation.location.y,
                     '.',markeredgewidth=lwidth,color=lcolor)
        elif isinstance(sampled_landmark.representation, LineRepresentation):
            xs = [sampled_landmark.representation.line.start.x,sampled_landmark.representation.line.end.x]
            ys = [sampled_landmark.representation.line.start.y,sampled_landmark.representation.line.end.y]
            plt.fill(xs,ys,facecolor='none',edgecolor=lcolor,linewidth=lwidth)
        elif isinstance(sampled_landmark.representation, RectangleRepresentation):
            rect = sampled_landmark .representation.rect
            xs = [rect.min_point.x,rect.min_point.x,rect.max_point.x,rect.max_point.x]
            ys = [rect.min_point.y,rect.max_point.y,rect.max_point.y,rect.min_point.y]
            plt.fill(xs,ys,facecolor='none',edgecolor=lcolor,linewidth=lwidth)


        # rel_scores = []
        # for relation in relations:
        #     rel_scores.append( relation.probability(poi, sampled_landmark) )
        # rel_scores = array(rel_scores)
        # rel_probabilities = rel_scores/sum(rel_scores)
        # index = rel_probabilities.cumsum().searchsorted( random.sample(1) )[0]
        # sampled_relation = relations[index]

        # toprint = str(poi)+' ; '+sampled_relation.get_description() + " " + sampled_landmark.get_description()
        # print toprint
        plt.axis('scaled')

        plt.axis([scene_bb.min_point.x, scene_bb.max_point.x, scene_bb.min_point.y, scene_bb.max_point.y])
        title = "Probability of location given description:\n" + description
        # plt.suptitle('\n'.join(wrap(title,50)))
        plt.suptitle(title)


        # plt.subplot(2,2,2)
        # plt.axis([0,1.5,-0.1,1.1])

        # xs = arange(0,3,0.01)
        # ys = {}
        # for relation in relations:
        #     name = relation.__class__.__name__
        #     plt.plot( xs, relation.distance_probability(xs) )
        # plt.axvline(x=distance,linewidth=2)
        # print distance


        plt.show()

# import sys
# if __name__ == '__main__':
#     if len(sys.argv) > 1:
#         howmany = int(sys.argv[1])
#     else:
#         howmany = 100
#     for i in range(howmany):
#         main()
