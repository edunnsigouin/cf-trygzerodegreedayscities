"""
hard-coded paths in trygzerodegreedayscities
"""

cf_space             = "/nird/datapeak/NS9873K/etdu/"
proj                 = "/nird/home/edu061/cf-trygzerodegreedayscities/"
data                 = proj + "data/"
fig                  = proj + "fig/"

raw                  = cf_space + 'raw/'
processed            = cf_space + 'processed/cf-trygzerodegreedayscities/'

senorge_raw          = '/nird/projects/NS9873K/DATA/senorge/'
senorge_processed    = processed + '/senorge/'

eobs_raw             = raw + 'eobs/daily/31_0e/'
eobs_processed       = processed + 'eobs/31_0e/'

dirs = {"senorge_processed":senorge_processed,
        "senorge_raw":senorge_raw,
        "eobs_processed":eobs_processed,
        "eobs_raw":eobs_raw,
        "fig":fig,
        "proj":proj,
        "data":data,
}


cities = ['Oslo','Kristiansand','Stavanger','Bergen','Ålesund','Trondheim','Bodø','Tromsø','Lillehammer','Alta']



