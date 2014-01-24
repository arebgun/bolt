import construction as struct
import lexical_items as li
import sempoles as sp



class MeasurePhrase(struct.Construction):
    pattern = [li.Measure]
    arg_indices = [0]
    function = sp.ReturnUnaltered

class DegreeMeasurePhrase(MeasurePhrase):
    pattern = [li.DegreeModifier, li.Measure]
    arg_indices = [0,1]
    function = sp.DegreeModify



# class CompoundRelation(struct.Construction, struct.Relation):
#     pass

class DistanceRelation(struct.Construction, struct.Relation):
    pattern = [MeasurePhrase, li.DestinationTag]
    arg_indices = [0]
    function = sp.DistanceRelate

class OrientationRelation(struct.Construction, struct.Relation):
    pattern = [li.DestinationTag, li.Article, li.Direction, li.BelongingTag]
    arg_indices = [2]
    function = sp.OrientationRelate

class PartOfRelation(struct.Construction, struct.Relation):
    pattern= [li.BelongingTag]
    arg_indices=[]
    function = sp.PartOfRelate


class OrientationAdjective(struct.Construction, li.Adjective):
    pattern = [li.Direction]
    arg_indices = [0]
    function = sp.OrientationAdjectify

class AdjectivePhrase(struct.Construction):
    pattern = [li.Adjective]
    arg_indices = [0]
    function = sp.ReturnUnaltered

class TwoAdjectivePhrase(AdjectivePhrase):
    pattern = [li.Adjective, li.Adjective]
    arg_indices = [0,1]
    function = sp.PropertyCombine

class DegreeAdjectivePhrase(AdjectivePhrase):
    pattern = [li.DegreeModifier, li.GradableAdjective]
    arg_indices = [0,1]
    function = sp.DegreeModify



class NounPhrase(struct.Construction):
    pattern = [li.Noun]
    arg_indices = [0]
    function = sp.ReturnUnaltered

class AdjectiveNounPhrase(NounPhrase):
    pattern = [AdjectivePhrase, li.Noun]
    arg_indices = [0,1]
    function = sp.PropertyCombine




class ReferringExpression(struct.Construction):
    pattern = [li.Article, NounPhrase]
    arg_indices = [0,1]
    function = sp.ArticleCombine

class RelationLandmarkPhrase(struct.Construction):
    pattern = [struct.Relation, ReferringExpression]
    arg_indices = [0,1]
    function = sp.RelateToLandmark

class RelationNounPhrase(struct.Construction):
    pattern = [NounPhrase, RelationLandmarkPhrase]
    arg_indices = [0,1]
    function = sp.NounPhraseRelate

class ExtrinsicReferringExpression(ReferringExpression):
    pattern = [li.Article, RelationNounPhrase]
    arg_indices = [0,1]
    function = sp.ArticleCombine



constructions_list = [
    AdjectivePhrase,
    DegreeAdjectivePhrase,
    NounPhrase,
    AdjectiveNounPhrase,
    MeasurePhrase,
    DegreeMeasurePhrase,
    DistanceRelation,
    OrientationRelation,
    ReferringExpression,
    RelationLandmarkPhrase,
    RelationNounPhrase,
    ExtrinsicReferringExpression
]