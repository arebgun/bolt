import sys
sys.path.insert(1,'..')
import automain
import argparse
import random
import shelve
import traceback
import random
import functools as ft
import multiprocessing as mp
import sqlalchemy as alc

import utils
import common as cmn
import semantics as sem
import language_user as lu
import lexical_items as li
import constructions as st
import partial_parser as pp
from all_parse_generator import AllParseGenerator as apg

import collections as coll
import time
import re
import operator as op
from matplotlib import pyplot as plt
from enchant.checker import SpellChecker
chkr = SpellChecker("en_US")

import IPython

t_lexicon = [
    li._,
    li.the,
    li.objct,
    li.block,
    # li.box,
    li.sphere,
    # li.ball,
    li.cylinder,
    li.table,
    li.corner,
    li.edge,
    li.end,
    li.half,
    li.middle,
    li.side,
    li.red,
    # li.orange,
    # li.yellow,
    li.green,
    li.blue,
    # li.purple,
    # li.pink,
    # li.black,
    # li.white,
    # li.gray,
    li.front,
    li.back,
    li.left,
    li.right,
    li.to,
    li.frm,
    li.of,
    li.far,
    li.near,
    li.on,

]

t_structicon = [
    st.OrientationAdjective,
    st.AdjectivePhrase,
    st.TwoAdjectivePhrase,
    st.DegreeAdjectivePhrase,
    st.NounPhrase,
    st.AdjectiveNounPhrase,
    st.MeasurePhrase,
    st.DegreeMeasurePhrase,
    st.PartOfRelation,
    st.DistanceRelation,
    st.OrientationRelation,
    st.ReferringExpression,
    st.RelationLandmarkPhrase,
    st.RelationNounPhrase,
    st.ExtrinsicReferringExpression,

    # st.ReferringExpression2,
    # st.ExtrinsicReferringExpression2,
]

# lexicon_words = [x.regex for x in t_lexicon]



extended_t_lexicon = [
    li._,
    li.the,
      li.a_article,
      li.an_article,

    li.objct,
    li.block,
    li.sphere,
    li.cylinder,
    li.table,
      li.cube_noun,
      li.ball,
      li.box,
      li.rectangle_noun,
      li.square_noun,
      li.shape_noun,
      li.circle_noun,
      li.prism_noun,
      li.cuboid_noun,

    li.corner,
    li.edge,
    li.end,
    li.half,
    li.middle,
    li.side,
      # li.center_noun,
      # li.top_noun,

    li.red,
    li.green,
    li.blue,
      li.purple,
      li.yellow,
      li.orange,
      li.brown_adj,
      li.colored_adj,
      li.pink,
      li.violet_adj,

      li.rectangular_adj,
      li.square_adj,
      li.round_adj,
      li.shaped_adj,
      li.circular_adj,

      # li.small_adj,
      # li.short_adj,
      # li.large_adj,

    li.front,
    li.back,
    li.left,
    li.right,
      li.near_dir,
      li.far_dir,
      li.top_dir,
    li.to,
    li.frm,
    li.of,

    li.far,
    li.near,
    li.on,
      li.in_,
      li.near_rel,
      li.at_rel,
      li.next_to_rel,
      li.behind_rel,
      li.in_front_of_rel,
      li.towards_rel,
      li.close_to_rel,
      li.above_rel,
      li.away_from_rel,
]

extended_t_structicon = [
    st.OrientationAdjective,
    st.AdjectivePhrase,
    st.TwoAdjectivePhrase,
    st.DegreeAdjectivePhrase,
    st.NounPhrase,
    st.AdjectiveNounPhrase,
    st.MeasurePhrase,
    st.DegreeMeasurePhrase,
    st.PartOfRelation,
    st.DistanceRelation,
    st.OrientationRelation,
    st.ReferringExpression,
    st.RelationLandmarkPhrase,
    st.RelationNounPhrase,
    st.ExtrinsicReferringExpression,

      # st.ReferringExpression2,
      # st.ExtrinsicReferringExpression2,
]

# extended_lexicon_words = [x.regex for x in extended_s_lexicon]

engine = alc.create_engine('sqlite:///mtbolt.db', echo=True)
meta = alc.MetaData()
meta.reflect(bind=engine)
entities = meta.tables['scenes_entity']
descriptions = meta.tables['tasks_descriptionquestion']

q = alc.sql.select([entities.c.scene_id, 
                    entities.c.name, 
                    descriptions.c.answer,
                    descriptions.c.object_description,
                    descriptions.c.location_description,
                    descriptions.c.use_in_object_tasks]).where(
                    entities.c.id==descriptions.c.entity_id)
conn = engine.connect()
results = list(conn.execute(q))
# results = [x for x in zip(*list(conn.execute(q)))[0] if len(x)>0]
conn.close()
scene_ids, entity_names, answers, objdescriptions, locdescriptions, turk_answered = zip(*results)

scene_infos = sem.run.read_scenes('static_scenes',True)
scenes = {}
reverse_scenes = {}
for num, scene, speaker in scene_infos:
    scenes[num] = (scene, speaker)
    reverse_scenes[scene] = num

answers, locdescriptions = list(answers), list(locdescriptions)
for i in range(len(answers)):
    if len(answers[i]) > 0:
        assert(len(locdescriptions[i]) == 0)
        locdescriptions[i] = answers[i]
    if len(locdescriptions[i]) == 0:
        assert(len(answers[i]) > 0)


# objCharCounts = coll.Counter()
# [objCharCounts.update(u) for u in objdescriptions]

def getMinWordFrequency(utterance, delimiters, counts):
    minF = float('inf')
    for word in re.split(delimiters,utterance.lower()):
        if len(word) > 0:
            if counts[word] < minF:
                minF = counts[word]
    return minF

original_oov = dict(
articles = ['a','an'],
location_words = ['center','top','bottom','upper','rear','lower','north'],
shape_nouns = ['cube','ball','box','rectangle','square','shape','circle',
               'prism','cuboid','disk','disc','puck','brick'],
shape_adjectives = ['rectangular','square','round','shaped','circular',
                    'cylindrical','spherical'],
color_adjectives = ['purple','yellow','orange','brown','colored','pink',
                    'violet','dark','black','light'],
dimensional_adjectives = ['small','short','large','tall','big','medium',
                          'flat','sized','little','upright','wide','tiny'],
relations = ['in','near','at','next to','between','behind','towards',
             'closest to','close to','above','away from','beside','below',
             'toward','by','adjacent to','before','opposite'],

# conjunctions = ['and', 'but'],
numbers = ['two','one','three','second','first','four','five','third','number'],
pronouns = ['other','it','all','its','that','me','this','another'],
# plurals = ['objects','cubes','sides','blocks','shapes','boxes','rectangles'],
verbs = ['is','placed','located','standing','positioned','sitting','looks','lying',
         'touching','has','laying','compared'],
adverbs = ['just','slightly','very','almost','about','directly','nearly','only'],
comparatives = ['closest','farthest','smallest','than','furthest','last','nearest','closer',
                'nearer','smaller','most','biggest','farther','largest','larger','more'],

unamerican = ['colour','centre','coloured'],
misc = ['color','with','hand','position','which','way','viewer','down','present','off',
        'bit','halfway','also','not','up','place','part','like','solid','piece',
        'dimensional','s','length','as'],
with_freq_20_or_less = ['colours','breadth','partial','pointing','represent','comparatively','focus','skin','go','chair','rectangles','children','row','whose','orb','contacted',"two's",'stacked','rightmost','thinnest','under','inwards','longways','southwest','fat','geometrical','smack','every','fall','fifths','exact','fag','posture','prefect','parrot','presented','wooden','die','duster','laying','colors','skewed','brownish','greenish','smallish','oblong','glued','almond','direct','past','surrounding','design','perspective','further','odd','even','what','stands','centimeter','section','4','thickness','version','falling','thin','body','led','degree','component','here','boll','let','others','alone','along','fifteen','appears','cubical','extreme','fatter','dividing','ahead','items','example','spheres','narrow','makes','midway','elongated','sticking','golden','named','adjoining','visible','boxy','verge','oval','forefront','clockwise','tools','landscape','lime','ash','would','mid','occupying','few','camera','music','card','type','start','sort','entrance','inclined','chocolate','opposing','surface','rare','aligned','dot','must','none','fourths','rights','cap','eclipse','labeled','nine','can','making','marble','something','beautiful','compare','figure','slim','share','northernmost','high','ellipsoid','tallish','numbers','magenta','viewers','tan','axis','huge','colorful','locates','sit','rather','means','arrangement','1','how','actual','fourth','butting','beige','yellowish','after','southern','tied','bothered','diagram','misshapen','such','horizontal','parallel','cuboids','lines','quadrant','upside','shade','enter','inline','lined','egg','order','cubs','six','rounded','backs','over','proportionate','ended','cyan','mentioned','diagonally','still','blocked','perfect','horizontally','group','apex','actually','according','degrees','dish','then','them','good','greater','yo','material','equidistant','they','foot','now','bigger','represents','name','lies','corners','level','faraway','each','found','beneath','quarter','tray',"isn't",'upward','heavy','tot','tilted','list','globe','inverted','prisms','hemispheric','precedes','our','beyond','extract','thick','out','shown','container','cartridge','space','rounding','shallow','surrounded','squared','forty','looking','pool','discounting','squares','forth','reg',"one's",'foreground','squished','sideways','midpoint','diagonal','free','quite','geared','base','besides','earliest','beginning','shortest','backwards','backside','widths','motion','thing','disturbed','onto','think','south','chopped','there','distal','scene','nested','feet','done','least','misshaped','array',"viewer's",'tuna','size','squashed','similarly','leading','diameter','approximately','their','2','hexagon','white','perfectly','structures','shifted','hue','option','centered','tool','exactly','took','immediate','lavender','western','somewhat','holder','distance','kind','double','board','glossy','were','posit','mini','turned','head','unlikely','have','ans','curved','hexagonal','slab','any','viewed','relatively','lid','sandwiched','squat','angle','form','able','aside','oriented','laterally','chess','so','lit','soap','play','though','glob','reach','fore','circled','device','coin','extremely',"object's",'average','face','points','typically','magnolia','fact','shot','cereal','gold','lengthwise','corned','cake','bright','relation','earth','rough','font','thirds','find','fifth','northeast','based','dice','northern','proportion','enough','dull','wood','hockey','pretty','watching','partially','centrally','plate','his','triangle','viewpoint','wide','dd','television','trees','foremost','longest','wise','gray',"ball's",'bar','caramel','common','x','rapped','fixed','where','vision','view','bars','set','observer','frame','ends','relative','differs','see','computer','radius','are','best','closer','said','closet','ways','straightly','topmost','3','various','closed','across','we','terms','vertical','nature','screen','however','southeast','magi','coffee','roller','both','drum','tablet','resides','many','barely','equal','against','facet','somewhere','s','figures','faces','exceeds','baseball','nearby','among','point','bricks','royal','narrower','height','anti','likes','learning','wider','respect','olives','tint','west','dusting','colorblind','tiny','much','reflective','lovely','hardly','empty','location','resting','partly','painted','fours','formed','boxed','else','finished','deep','angles','obscuring','those','myself','east','tank','look','these','isolated','straight','air','ugly','while','leaning','furthermost','pack','site','leans','iv','cluster',"cylinder's",'itself','sits','conical','ring','if','different','dab','miniature','lengths','make','same','shorter','parts','speaker','northwest','inch','cubed','several','showing','higher','wheel','fairly','pinkish','used','cubic','nest','nestled','edges','moving','user','assortment','numbered','vantage','shaded','kept','database','i','well','roughly','person','without','bottle','y','dimension','taller','stout','less','being','photo','proximity','rotated','rest','quarters','nearing','violent','distant','positions','touch','yes','previous','tables','sandal','widest','rose','percent','seems','except','seem','chartreuse','combine','attractive','easy','sweet','remaining','match','around','easily','couple','possibly','background','desk','clustered','immediately','unique','burnt','apart','vertically','d','gift','sizes','nude','plane',"table's",'facing','audience','t','fully','simply','tower','pillar','grouping','old','sequence','mauve','some','dead','printer','sight','pale','saucer','slice','dimensions','for','broad','tube','squire','everything','unit','leftmost','pea','locate','be','slight','perpendicular','offset','imaginary','central','getting','column','o','hinge','stand','shiny','or','raised','image','locus','nothing','been','lastly','your','custard','rows','her','area','placing','approximate','width','low','forward','was','lowest','building','cricket','complete','inches','forming','striker','game','line','highest','violate','he','reset','made','cylinders','places','tallest','attached','us',"rectangle's",'grasp','similar','called','rust','darker','separated','constant','angled','int','describe','am','pie','maroon','as','tabletop','ink','peach','no','when','portion','grouped','book','5','really','stumpy','roll','picture','mustard','accidental','separate','sided','vertex','variation','includes','meaning','flattened','kinda','rolling','velvet','structure','upwards','semicircle','longer','depth','lying','resembles','corer','yell','starting','having']
)

oov_categories = ['relations','location_words','shape_nouns','shape_adjectives','color_adjectives',
'dimensional_adjectives','articles','verbs','adverbs','pronouns',
'numbers','comparatives','unamerican','misc','with_freq_20_or_less']


def print_word_counts():
    objOovCounts = coll.defaultdict(coll.Counter)
    for u in objdescriptions:
        u = u.lower()
        for c in oov_categories:
            for w in original_oov[c]:
                if re.search(r'\b%s\b'%w,u) is not None:
                    objOovCounts[c][w]+=1
            # if any([re.search(r'\b%s\b'%w,u) is not None for w in original_oov[c]]):
            #     objOovCounts['has_oov_'+c]+=1

    locOovCounts = coll.defaultdict(coll.Counter)
    for u in locdescriptions:
        u = u.lower()
        for c in oov_categories:
            for w in original_oov[c]:
                if re.search(r'\b%s\b'%w,u) is not None:
                    locOovCounts[c][w]+=1
            # if any([re.search(r'\b%s\b'%w,u) is not None for w in original_oov[c]]):
            #     locOovCounts['has_oov_'+c]+=1

    longest = max([len(w) for c in oov_categories for w in original_oov[c]])
    for c in oov_categories:
        print c
        original_oov[c] = sorted(original_oov[c],key=lambda w:objOovCounts[c][w]+locOovCounts[c][w], reverse=True)
        for w in original_oov[c]:
            print '  ',w.rjust(longest),str(objOovCounts[c][w]).ljust(4),str(locOovCounts[c][w]).ljust(4),str(objOovCounts[c][w]+locOovCounts[c][w]).ljust(5),objOovCounts[c][w]+locOovCounts[c][w]>=100
        print



modified_oov = dict(
articles = [],
location_words = ['bottom','upper','rear','lower','north'],
shape_nouns = ['disk','disc','puck','brick'],
shape_adjectives = ['cylindrical','spherical'],
color_adjectives = ['dark','black','light'],
dimensional_adjectives = ['tall','big','medium',
                          'flat','sized','little','upright','wide','tiny'],
relations = ['between','closest to','beside','below',
             'toward','by','adjacent to','before','opposite'],

# conjunctions = ['and', 'but'],
numbers = ['two','one','three','second','first','four','five','third','number'],
pronouns = ['other','it','all','its','that','me','this','another'],
# plurals = ['objects','cubes','sides','blocks','shapes','boxes','rectangles'],
verbs = ['is','placed','located','standing','positioned','sitting','looks','lying',
         'touching','has','laying','compared'],
adverbs = ['just','slightly','very','almost','about','directly','nearly','only'],
comparatives = ['closest','farthest','smallest','than','furthest','last','nearest','closer',
                'nearer','smaller','most','biggest','farther','largest','larger','more'],

unamerican = ['colour','centre','coloured'],
misc = ['color','with','hand','position','which','way','viewer','down','present','off',
        'bit','halfway','also','not','up','place','part','like','solid','piece',
        'dimensional','s','length','as'],
with_freq_20_or_less = ['colours','breadth','partial','pointing','represent','comparatively','focus','skin','go','chair','rectangles','children','row','whose','orb','contacted',"two's",'stacked','rightmost','thinnest','under','inwards','longways','southwest','fat','geometrical','smack','every','fall','fifths','exact','fag','posture','prefect','parrot','presented','wooden','die','duster','laying','colors','skewed','brownish','greenish','smallish','oblong','glued','almond','direct','past','surrounding','design','perspective','further','odd','even','what','stands','centimeter','section','4','thickness','version','falling','thin','body','led','degree','component','here','boll','let','others','alone','along','fifteen','appears','cubical','extreme','fatter','dividing','ahead','items','example','spheres','narrow','makes','midway','elongated','sticking','golden','named','adjoining','visible','boxy','verge','oval','forefront','clockwise','tools','landscape','lime','ash','would','mid','occupying','few','camera','music','card','type','start','sort','entrance','inclined','chocolate','opposing','surface','rare','aligned','dot','must','none','fourths','rights','cap','eclipse','labeled','nine','can','making','marble','something','beautiful','compare','figure','slim','share','northernmost','high','ellipsoid','tallish','numbers','magenta','viewers','tan','axis','huge','colorful','locates','sit','rather','means','arrangement','1','how','actual','fourth','butting','beige','yellowish','after','southern','tied','bothered','diagram','misshapen','such','horizontal','parallel','cuboids','lines','quadrant','upside','shade','enter','inline','lined','egg','order','cubs','six','rounded','backs','over','proportionate','ended','cyan','mentioned','diagonally','still','blocked','perfect','horizontally','group','apex','actually','according','degrees','dish','then','them','good','greater','yo','material','equidistant','they','foot','now','bigger','represents','name','lies','corners','level','faraway','each','found','beneath','quarter','tray',"isn't",'upward','heavy','tot','tilted','list','globe','inverted','prisms','hemispheric','precedes','our','beyond','extract','thick','out','shown','container','cartridge','space','rounding','shallow','surrounded','squared','forty','looking','pool','discounting','squares','forth','reg',"one's",'foreground','squished','sideways','midpoint','diagonal','free','quite','geared','base','besides','earliest','beginning','shortest','backwards','backside','widths','motion','thing','disturbed','onto','think','south','chopped','there','distal','scene','nested','feet','done','least','misshaped','array',"viewer's",'tuna','size','squashed','similarly','leading','diameter','approximately','their','2','hexagon','white','perfectly','structures','shifted','hue','option','centered','tool','exactly','took','immediate','lavender','western','somewhat','holder','distance','kind','double','board','glossy','were','posit','mini','turned','head','unlikely','have','ans','curved','hexagonal','slab','any','viewed','relatively','lid','sandwiched','squat','angle','form','able','aside','oriented','laterally','chess','so','lit','soap','play','though','glob','reach','fore','circled','device','coin','extremely',"object's",'average','face','points','typically','magnolia','fact','shot','cereal','gold','lengthwise','corned','cake','bright','relation','earth','rough','font','thirds','find','fifth','northeast','based','dice','northern','proportion','enough','dull','wood','hockey','pretty','watching','partially','centrally','plate','his','triangle','viewpoint','wide','dd','television','trees','foremost','longest','wise','gray',"ball's",'bar','caramel','common','x','rapped','fixed','where','vision','view','bars','set','observer','frame','ends','relative','differs','see','computer','radius','are','best','closer','said','closet','ways','straightly','topmost','3','various','closed','across','we','terms','vertical','nature','screen','however','southeast','magi','coffee','roller','both','drum','tablet','resides','many','barely','equal','against','facet','somewhere','s','figures','faces','exceeds','baseball','nearby','among','point','bricks','royal','narrower','height','anti','likes','learning','wider','respect','olives','tint','west','dusting','colorblind','tiny','much','reflective','lovely','hardly','empty','location','resting','partly','painted','fours','formed','boxed','else','finished','deep','angles','obscuring','those','myself','east','tank','look','these','isolated','straight','air','ugly','while','leaning','furthermost','pack','site','leans','iv','cluster',"cylinder's",'itself','sits','conical','ring','if','different','dab','miniature','lengths','make','same','shorter','parts','speaker','northwest','inch','cubed','several','showing','higher','wheel','fairly','pinkish','used','cubic','nest','nestled','edges','moving','user','assortment','numbered','vantage','shaded','kept','database','i','well','roughly','person','without','bottle','y','dimension','taller','stout','less','being','photo','proximity','rotated','rest','quarters','nearing','violent','distant','positions','touch','yes','previous','tables','sandal','widest','rose','percent','seems','except','seem','chartreuse','combine','attractive','easy','sweet','remaining','match','around','easily','couple','possibly','background','desk','clustered','immediately','unique','burnt','apart','vertically','d','gift','sizes','nude','plane',"table's",'facing','audience','t','fully','simply','tower','pillar','grouping','old','sequence','mauve','some','dead','printer','sight','pale','saucer','slice','dimensions','for','broad','tube','squire','everything','unit','leftmost','pea','locate','be','slight','perpendicular','offset','imaginary','central','getting','column','o','hinge','stand','shiny','or','raised','image','locus','nothing','been','lastly','your','custard','rows','her','area','placing','approximate','width','low','forward','was','lowest','building','cricket','complete','inches','forming','striker','game','line','highest','violate','he','reset','made','cylinders','places','tallest','attached','us',"rectangle's",'grasp','similar','called','rust','darker','separated','constant','angled','int','describe','am','pie','maroon','as','tabletop','ink','peach','no','when','portion','grouped','book','5','really','stumpy','roll','picture','mustard','accidental','separate','sided','vertex','variation','includes','meaning','flattened','kinda','rolling','velvet','structure','upwards','semicircle','longer','depth','lying','resembles','corer','yell','starting','having']
)

def replace_all(string, patterns):
    for pattern, replacement in patterns:
        string = re.sub(r'\s*\b%s\b\s*' % pattern,replacement,string)
    return string

def get_error_counts(descriptions, scene_ids, entity_names, turk_answered, 
                     oov_dict, lexicon, structicon, parse_goal,
                     a_okay=False, split_parse=False, to_replace=None):
    # delimiters = " |&|\(|\)|-|,|\/|\.|\?|;|>"
    # wordCounts = coll.Counter()
    # [wordCounts.update(re.split(delimiters,u.lower())) for u in objdescriptions]
    # print wordCounts
    # IPython.embed()
    # exit()

        

    parser = lu.LanguageUser(name='Parser', lexicon=lexicon, 
                             structicon=structicon, meta=None, remember=False)
    unparseable = []
    parseable = []

    def parse(utterance):
        parses = list(parser.parse(utterance, max_holes=0))
        return parses

    punct_re = "'|&|\(|\)|-|,|\/|\.|\?|;|>"
    # numb_re = "1|2|3|4|5"
    articles = ['a','an','the']
    conjunctions = ['and', 'but', 'or']
    plurals = ['objects','cubes','sides','blocks','shapes','boxes','rectangles']

    error_counts = coll.Counter()

    # oovs = coll.Counter()

    for original,s,e,ta in zip(descriptions,scene_ids,entity_names,turk_answered):
        original = original.strip()
        scene, speaker = scenes[s]
        entity = (scene.landmarks['object_'+e],)
        if len(original) == 0:
            continue

        if to_replace is not None:
            u = replace_all(original,to_replace)
        else:
            u = original

        error_counts['total']+=1

        syntax_error = False

        u2 = u.strip('.').strip('.').strip('.').strip('?')
        if u2 != u:
            syntax_error = True
            if re.split(punct_re, u2)[0] == u2:
                error_counts['has_final_punctuation_only']+=1
            else:
                error_counts['has_middle_punctuation']+=1
        elif re.split(punct_re, u)[0] != u:
            error_counts['has_middle_punctuation']+=1
            syntax_error = True

        u = u.lower()

        if any([re.search(r'\b%s\b'%w,u) is not None for w in conjunctions]):
            error_counts['has_conjunctions']+=1
            syntax_error = True
        if any([re.search(r'\b%s\b'%w,u) is not None for w in plurals]):
            error_counts['has_plurals']+=1
            syntax_error = True
        chkr.set_text(u)
        if len([err for err in chkr]) > 0:
            error_counts['has_spelling_error']+=1
            syntax_error = True
        if not any([re.search(r'\b%s\b'%w,u) is not None for w in articles]):
            error_counts['has_no_articles']+=1
            if not a_okay:
              syntax_error = True

        if syntax_error:
            error_counts['has_syntax_issue']+=1

        vocab_error = False

        for c in oov_categories:
            if any([re.search(r'\b%s\b'%w,u) is not None for w in oov_dict[c]]):
                error_counts['has_oov_'+c]+=1
                vocab_error = True


        if vocab_error:
            error_counts['has_vocab_issue']+=1

        if syntax_error or vocab_error:
            error_counts['has_known_issue']+=1

        to_split = r'\.|,|;|\band\b|\bbut\b|\bor\b'
        if split_parse and re.search(to_split,u) is not None:
            u_parseable = []
            for phrase in re.split(to_split,u):
                parses = parse(phrase)
                if len(parses) > 0 and isinstance(parses[0].current[0],parse_goal):
                    u_parseable.append(parses[0].current[0])
            if len(u_parseable) > 0:
                parseable.append((scene, speaker, (entity,original,u,ta,u_parseable)))
            else:
                unparseable.append((scene, speaker, (entity,original,u,ta,[])))
                error_counts['reagent_cant_parse']+=1
        else:
            parses = parse(u)
            if len(parses) > 0 and isinstance(parses[0].current[0],parse_goal):
                parseable.append((scene, speaker, (entity,original,u,ta,[parses[0].current[0]])))
            else:
                error_counts['reagent_cant_parse']+=1
                unparseable.append((scene, speaker, (entity,original,u,ta,[])))

        # else:
        #     print u

    return error_counts, parseable, unparseable

error_categories = [

'has_oov_articles',
'has_oov_location_words',
'has_oov_shape_nouns',
'has_oov_shape_adjectives',
'has_oov_color_adjectives',
'has_oov_dimensional_adjectives',
'has_oov_relations',
'has_oov_numbers',
'has_oov_pronouns',
'has_oov_verbs',
'has_oov_adverbs',
'has_oov_comparatives',
'has_oov_unamerican',
'has_oov_misc',
'has_oov_with_freq_20_or_less',
'has_vocab_issue',

'has_conjunctions',
'has_plurals',
'has_middle_punctuation',
'has_final_punctuation_only',
'has_spelling_error',
'has_no_articles',
'has_syntax_issue',

'has_known_issue',
'reagent_cant_parse'

]
# plt.ion()
def print_error_counts(error_counts):
    total = float(error_counts['total'])
    longest = max([len(ec) for ec in error_categories])
    for ec in error_categories:
        print ec.rjust(longest),
        print str(error_counts[ec]).ljust(4),
        print '%0.2f' % (error_counts[ec]/total)
    print

import numpy as np
# def plot_error_counts(error_counts):
#     total = float(error_counts['total'])
#     ind = np.array(range(16)+range(17,24)+range(25,27))
#     heights = np.array([error_counts[ec] for ec in error_categories])/total
#     plt.barh(ind,heights)
#     plt.subplots_adjust(left=0.35)
#     plt.yticks(ind+0.5,error_categories)
#     plt.ylim((0,ind[-1]+1))
#     plt.xlim((0,1))
#     plt.show()


def plot_error_counts(error_counts, title):
    total = float(error_counts['total'])
    ind = np.array(range(2)+range(3,10)+range(11,27))
    heights = np.array([error_counts[ec] for ec in reversed(error_categories)])/total

    fig, ax = plt.subplots()
    ax.set_ylim((0,ind[-1]+1))
    ax.set_xlim((0,1))
    fig.subplots_adjust(left=0.35)
    ax.set_yticks(ind+0.5)
    ax.set_yticklabels(list(reversed(error_categories)))
    
    ax.barh(ind,heights)

    fig.show()

def plot_before_after(before_counts, after_counts, title):
    total = float(before_counts['total'])
    ind = np.array(range(2)+range(3,10)+range(11,27))
    heights = np.array([before_counts[ec] for ec in reversed(error_categories)])/total
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.35)
    ax.set_yticks(ind+0.5)
    ax.set_yticklabels(list(reversed(error_categories)))
    ax.set_ylim((0,ind[-1]+1))
    ax.set_xlim((0,1))
    ax.set_xlabel('Fraction of descriptions (out of %i)' % total)

    ax.barh(ind,heights,color=(.35,.6,.9),label='Before fixes')

    total = float(after_counts['total'])
    ind = np.array(range(2)+range(3,10)+range(11,27))
    heights = np.array([after_counts[ec] for ec in reversed(error_categories)])/total

    ax.barh(ind,heights,color=(.9,.6,.2),label='After fixes')

    ax.set_title(title)
    ax.legend()
    ax.grid(axis='x')

    fig.show()

def lstrip(pattern, string):
    if string.startswith(pattern):
        string = string[len(pattern):].strip()
    return string



to_remove = ['it']+original_oov['verbs']+original_oov['adverbs']+original_oov['dimensional_adjectives']
to_remove = list(zip(to_remove,[' ']*len(to_remove)))

# to_replace = [('centre',' center '),('colour',' color ')]

descriptions_ = objdescriptions
parse_goal = st.ReferringExpression
# descriptions_ = locdescriptions
# parse_goal = st.RelationLandmarkPhrase
print
print 'Object Descriptions'
print 'Initial'
oerror_counts1, oparseable, ounparseable = get_error_counts(descriptions_, scene_ids, entity_names, turk_answered, original_oov, t_lexicon, t_structicon, parse_goal)
# print_error_counts( oerror_counts1 )
print 'Stripped Punctuation, Removed Verbs and Adverbs'
stripped = [u.lower().strip().strip('?').strip().strip('.').strip().strip('.').strip().strip('.').strip() for u in descriptions_]
# oerror_counts2, oparseable, ounparseable = get_error_counts(stripped, modified_oov, t_lexicon, t_structicon, parse_goal, a_okay=True)
# print_error_counts( oerror_counts2 )
print 'Extended Vocab'
# oerror_counts3, oparseable, ounparseable = get_error_counts(stripped, modified_oov, extended_t_lexicon, t_structicon, parse_goal)
# print_error_counts( oerror_counts3 )
# print 'No article allowed'
# oerror_counts4, oparseable, ounparseable = get_error_counts(stripped, modified_oov, extended_t_lexicon, extended_t_structicon, parse_goal, a_okay=True)
# print_error_counts( oerror_counts4 )
# print 'Americanized'
print 'Splitting'
# americanized = [re.sub('centre','center',re.sub('colour','color',u)) for u in stripped]
oerror_counts5, oparseable, ounparseable = get_error_counts(stripped, scene_ids, entity_names, turk_answered,  modified_oov, extended_t_lexicon, extended_t_structicon, parse_goal, split_parse=True, to_replace=to_remove)
print_error_counts( oerror_counts5 )
plot_before_after( oerror_counts1, oerror_counts5, 'Turk Object Descriptions Parsing Breakdown')

# IPython.embed()

descriptions_ = locdescriptions
parse_goal = st.RelationLandmarkPhrase
print
print 'Location Descriptions'
print 'Initial'
lerror_counts1, lparseable, lunparseable = get_error_counts(descriptions_, scene_ids, entity_names, turk_answered, original_oov, t_lexicon, t_structicon, parse_goal)
print_error_counts( lerror_counts1 )
print 'Stripped Punctuation and Verbs'
stripped = [u.lower().strip().strip('?').strip().strip('.').strip().strip('.').strip().strip('.').strip() for u in descriptions_]
# lerror_counts2, lparseable, lunparseable = get_error_counts(stripped, modified_oov, extended_t_lexicon, extended_t_structicon, parse_goal, a_okay=True)
# print_error_counts( lerror_counts2 )
print 'Extended Vocab'
# lerror_counts3, lparseable, lunparseable = get_error_counts(stripped, modified_oov, extended_t_lexicon, extended_t_structicon, parse_goal)
# print_error_counts( lerror_counts3 )
# print 'No article allowed'
# lerror_counts4, lparseable, lunparseable = get_error_counts(stripped, modified_oov, extended_t_lexicon, extended_t_structicon, parse_goal, a_okay=True)
# print_error_counts( lerror_counts4 )
# print 'Americanized'
# americanized = [re.sub('centre','center',re.sub('colour','color',u)) for u in stripped]
print 'Splitting'
lerror_counts5, lparseable, lunparseable = get_error_counts(stripped, scene_ids, entity_names, turk_answered, modified_oov, extended_t_lexicon, extended_t_structicon, parse_goal, split_parse=True, to_replace=to_remove)
print_error_counts( lerror_counts5 )
plot_before_after( lerror_counts1, lerror_counts5, 'Turk Location Descriptions Parsing Breakdown')

def write_csv(filename,l):
    with open(filename, 'w') as f:
        for x, ps in l:
            for i in x:
                f.write('%s|'%str(i))
            for p in ps:
                f.write('%s|' %repr(p.prettyprint()))    
            f.write('\r\n')

# def write_lines(filename,l):
#     with open(filename, 'w') as f:
#         for u in l:
#             f.write('%s\r\n'%u)

# oparseable_shuffled = [((reverse_scenes[sc],e,o,m),ps) for sc,sp,(e,o,m,b,ps) in oparseable]
# random.seed(0)
# random.shuffle(oparseable_shuffled)
# write_csv('object_parseable.csv',oparseable_shuffled)

# ounparseable_shuffled = [((reverse_scenes[sc],e,o,m),ps) for sc,sp,(e,o,m,b,ps) in ounparseable]
# random.seed(0)
# random.shuffle(ounparseable_shuffled)
# write_csv('object_unparseable.csv',ounparseable_shuffled)

# lparseable_turked = [((reverse_scenes[sc],e,o,m),ps) for sc,sp,(e,o,m,b,ps) in lparseable if b]
# random.seed(0)
# random.shuffle(lparseable_turked)
# write_csv('location_parseable_turked.csv',lparseable_turked)

# lparseable_unturked = [((reverse_scenes[sc],e,o,m),ps) for sc,sp,(e,o,m,b,ps) in lparseable if not b]
# random.seed(0)
# random.shuffle(lparseable_unturked)
# write_csv('location_parseable_unturked.csv',lparseable_unturked)

# lunparseable_turked = [((reverse_scenes[sc],e,o,m),ps) for sc,sp,(e,o,m,b,ps) in lunparseable if b]
# random.seed(0)
# random.shuffle(lunparseable_turked)
# write_csv('location_unparseable_turked.csv',lunparseable_turked)

# lunparseable_unturked = [((reverse_scenes[sc],e,o,m),ps) for sc,sp,(e,o,m,b,ps) in lunparseable if not b]
# random.seed(0)
# random.shuffle(lunparseable_unturked)
# write_csv('location_unparseable_unturked.csv',lunparseable_unturked)

# f = shelve.open('turk_intrinsic_training.shelf')
# f['turk'] = True
# f['seed'] = 0
# f['extrinsic'] = False
# f['training_data'] = [(sc,sp,[(e,ps)]) for sc,sp,(e,o,m,b,ps) in oparseable]
# f.close()

# f = shelve.open('turk_extrinsic_training.shelf')
# f['turk'] = True
# f['seed'] = 0
# f['extrinsic'] = True
# f['training_data'] = [(sc,sp,[(e,ps)]) for sc,sp,(e,o,m,b,ps) in lparseable if not b]
# f['test_data'] = [(sc,sp,[(e,ps)]) for sc,sp,(e,o,m,b,ps) in lparseable if b]
# f['bad_test_data'] = [(sc,sp,[(e,ps)]) for sc,sp,(e,o,m,b,ps) in lunparseable if b]
# f.close()

IPython.embed()
exit()