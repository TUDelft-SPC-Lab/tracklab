from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="data/SoccerNetMOT")
mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train", "test", "valid", "challenge"])

# After downloading the datasets, extract the data via # unzip -qq file.zip
# Then move the folders one level up