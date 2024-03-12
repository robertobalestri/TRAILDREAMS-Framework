from imdb import Cinemagoer
import requests
from bs4 import BeautifulSoup
import re
from common import get_video_file_details

class MovieInfo:
    """
    This class is used to store information about a movie.
    """
    ia = Cinemagoer('https', 'en')
    
    def __init__(self, code, movie_file_path):
        self.code = code
        self.directors = ""
        self.genres = []
        self.title = ""
        self.synopsis = ""
        self.quotes = []
        self.release_date = ""
        self.is_black_and_white = False
        self.file_duration, self.file_fps = get_video_file_details(movie_file_path)
        self.movie_file_path = movie_file_path
    
    def fetch_quotes_from_imdb(self, movie):
        url = self.ia.get_imdbURL(movie)
        quotes_url = f'{url}quotes/'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(quotes_url, headers=headers)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        quote_elements = soup.find_all(class_='ipc-list-card--border-line')
        raw_quotes = []
        for element in quote_elements:
            quotes = element.find_all('li')
            for quote in quotes:
                cleaned_quote = re.sub(r'<.*?>', '', quote.get_text(strip=True))
                raw_quotes.append(cleaned_quote)
        return raw_quotes
    
    @staticmethod
    def get_movie_synopsis(movie, index=0):
        synopsis_list = movie.get('synopsis', [])
        return synopsis_list[index] if synopsis_list else None

    def fill_with_imdb_info(self):
        try:
            movie = self.ia.get_movie(self.code)
            
            #print(self.ia.get_movie_infoset())
            
            #print(movie.infoset2keys)
                
            print(movie.get('color info', ''))
            print(movie.get('original air date', ''))
            print(movie.get('kind', ''))
            
            self.title = movie.get('title', 'N/A')
            print(self.title)
            self.directors = ', '.join([director['name'] for director in movie.get('directors', [])])
            print(self.directors)
            self.genres = movie.get('genres', [])
            print(str(self.genres))
            self.synopsis = self.get_movie_synopsis(movie)
            #print(self.synopsis)
            self.quotes = self.fetch_quotes_from_imdb(movie)
            #for quote in self.quotes:
                #print(quote)
            self.release_date = movie.get('original air date', 'N/A')
            
            
            if 'Color' in movie.get('color info', '')[0]:
                self.is_black_and_white = False
                print("Color")
            else:
                self.is_black_and_white = True
                
        except Exception as e:
            print(f"An error occurred: {e}")
        
        return self  # Return the instance itself

    # Getters
    def get_title(self):
        return self.title

    def get_directors(self):
        return self.directors

    def get_genres(self):
        return self.genres

    def get_synopsis(self):
        return self.synopsis

    def get_quotes(self):
        return self.quotes
    
    def get_release_date(self):
        return self.release_date
    
    def get_is_black_and_white(self):
        return self.is_black_and_white

    def get_movie_file_duration(self):
        return self.file_duration
    
    def get_movie_file_fps(self):
        return self.file_fps
    
    def get_movie_file_path(self):
        return self.movie_file_path
    
    def is_documentary(self):
        if 'Documentary' in self.genres:
            return True