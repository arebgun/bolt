import sys
sys.path.insert(1,'..')
import IPython
import common as cmn
import construction as con
import constructions as st
import lexical_items as li

a = st.ExtrinsicReferringExpression([
        li.the,
        st.RelationNounPhrase([
            st.NounPhrase([li.objct]),
            st.RelationLandmarkPhrase([
                st.OrientationRelation([
                    li.to,
                    li.the,
                    li.front,
                    li.of
                ]),
                st.ReferringExpression([
                    li.the,
                    st.NounPhrase([li.table])
                ])
            ])
        ])
    ])

b = st.ExtrinsicReferringExpression([
        li.the,
        st.RelationNounPhrase([
            st.NounPhrase([li.objct]),
            st.RelationLandmarkPhrase([
                cmn.Hole(con.Relation,
                [
                    li.to,
                    li.the,
                    li.front,
                    li.of
                ]),
                st.ReferringExpression([
                    li.the,
                    st.NounPhrase([li.table])
                ])
            ])
        ])
    ])

IPython.embed()