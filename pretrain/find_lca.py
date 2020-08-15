
from tqdm import tqdm
import pickle
import os

# open child2ar dict
f = open('/hpf/projects/brudno/marta/mimic_rs_collection/child2par.pickle', 'rb')
child2par = pickle.load(f)
f.close()
f = open("/hpf/projects/brudno/marta/mimic_rs_collection/allacronyms_meta2name_20190617.pickle", 'rb')
meta2name = pickle.load(f)
f.close()
f = open("/hpf/projects/brudno/marta/mimic_rs_collection/allacronyms_meta2cui_20190617.pickle", 'rb')
meta2cui = pickle.load(f)
f.close()

f = open("merge_expansions_20191031_final.pickle", 'rb')
merged_exps = pickle.load(f)
f.close()


def get_cui(name2cui, term):
  
    cui = sorted(name2cui[term])[-1]
    assert cui[0] == "C"
    return cui

#def get_name(cui):
 #   return cui2name[cui]

def get_ancestors(term, cui_list):
    term_ancestors = [term]
    visited = [term]
    seen = set()
    while visited:    
        curr_node = visited[0]
        visited = visited[1:]
        if curr_node not in child2par:
            continue
        if curr_node in seen:
            continue
        child2par[curr_node] = [ancestor for ancestor in child2par[curr_node] if ancestor in cui_list]
        #print(child2par[curr_node])
        ancestors = sorted(child2par[curr_node])
        term_ancestors.extend(ancestors)
        visited.extend(ancestors)
        seen.add(curr_node)
    return term_ancestors


def find_lca(abbr):
    # take input file
    name2cui = {}
    cui2name = {}
    
    cui_list = set()
    print("2")
    src_dir = "/hpf/projects/brudno/marta/mimic_rs_collection/cuis_rs_1000Samples_20190612"
    for root, subdir, filenames in os.walk(src_dir):
        print("3")
        for filename in tqdm(filenames):
            if "_1000.txt" in filename:
                cui = filename.split("_")[0]
                cui_terms = set()
                cui_list.add(cui)
                assert cui[0] == "C"
                with open(os.path.join(root, filename)) as fhandle:
                    for line in fhandle:
                        term = line.split("|")[0]
                        cui_terms.add(term)
                #if 'intravenous fluid therapy' in cui_terms:
                 #   print("found!", cui_terms, cui); import sys; sys.exit(0)
                #term = sorted(cui_terms)[0]
                cui2name[cui] = list(cui_terms)
              
                for term in sorted(cui_terms):
                    if term not in name2cui: 
                        name2cui[term] = set()
                    name2cui[term].add(cui)
    

    for abbr_ in meta2cui:
        for meta in meta2cui[abbr_]:
            # 0 : C
            cui = sorted(meta2cui[abbr_][meta])[0]
            if len(cui) == 1: continue
            assert cui[0] == "C", meta2cui[abbr_]
            
            ###cui = sorted(meta2cui[abbr_][meta])[0]
            possible_names = meta2name[abbr_][meta]
            name = ""
            ### for possible_name in possible_names:
            for possible_name in sorted(possible_names):
                ### if possible_name in merged_exps[abbr_]:
                if possible_name in sorted(merged_exps[abbr_]):
                    name = possible_name
                    break
            if name != "":
                name2cui[name] = [cui]
               # cui2name[cui] = [name]
                if cui not in cui2name:
                    cui2name[cui] = []
                cui2name[cui].append(name)
    
 


   # src_dir = "organized_close_medical_concepts_by_abbr_dist2.6_20200409_CASIONLY"
    src_dir = "organized_close_medical_concepts_by_abbr_dist2.6_20200809_cleaned"
    fname = "{}/{}.txt".format(src_dir, abbr)

    # each line is expansion:::relative
    exp2ancestors = {}

    concept2ancestors = {}
    with open(fname) as fhandle:
        for line in tqdm(fhandle):
            content = line[:-1].split(":::")
            expansion = ' '.join(content[0].split('_'))
            try:
                expansion_cui = get_cui(name2cui, expansion)
     
                relative = ' '.join(content[1].split('_'))
                relative_cui = get_cui(name2cui, relative)
            except:
                continue
            if expansion_cui not in exp2ancestors:
                exp2ancestors[expansion_cui] = get_ancestors(expansion_cui, cui_list)
            ancestors_expansion = exp2ancestors[expansion_cui]
            ancestors_relative = get_ancestors(relative_cui, cui_list)
            
            if expansion_cui not in concept2ancestors:
                concept2ancestors[expansion_cui] = set()
                concept2ancestors[expansion_cui].add(expansion_cui)
            if relative_cui not in concept2ancestors:
                concept2ancestors[relative_cui] = set()
                concept2ancestors[relative_cui].add(relative_cui)
            # check if overlap between expansion ancestors and relative ancesetors; find deepest ancestor for expansion
            lca = None
             
            for ancestor in ancestors_expansion:
                if ancestor in ancestors_relative:
                    #lca = sorted(get_name(ancestor))[0]
                    lca = ancestor
                    break
            # if lca is not None, add as ancestor for expansion and relative
            if lca:
                concept2ancestors[expansion_cui].add(lca)
                concept2ancestors[relative_cui].add(lca)
                if lca not in concept2ancestors:
                    concept2ancestors[lca] = set()
                    concept2ancestors[lca].add(lca)
    # check if concepts in dictionary are also ancestors to any other concepts; if so, link concepts
    for concept in tqdm(concept2ancestors):
        #concept_cui = get_cui(concept)
        concepts_ancestors = get_ancestors(concept, cui_list)
        for ancestor in concepts_ancestors:
            #ancestor = sorted(get_name(ancestor_cui))[0]
            if ancestor != concept and ancestor in concept2ancestors:
                concept2ancestors[concept].add(ancestor)

    dest_dir = "pickle_files"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    f = open("{}/child2ancs_{}_d2.6.pickle".format(dest_dir, abbr), 'wb')
    pickle.dump(concept2ancestors, f)
    f.close()
    print(concept2ancestors)

#find_lca('cea')
