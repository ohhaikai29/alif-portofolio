"""
Multi-Platform Sentiment Scraper
Author: Alif Octrio
Description: Automated sentiment extraction from YouTube, Reddit, X (Twitter), and TikTok
Security: All API credentials loaded from environment variables
"""

import os
import sys
import pandas as pd
import praw
from googleapiclient.discovery import build
from typing import List, Dict, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentScraper:
    """
    Multi-platform web scraper for sentiment analysis.
    Supports YouTube, Reddit, X (Twitter), and TikTok.
    """
    
    def __init__(self):
        """Initialize scraper with API credentials from environment variables."""
        self._validate_environment()
        self._initialize_clients()
    
    def _validate_environment(self):
        """Validate that all required environment variables are set."""
        required_vars = [
            'YOUTUBE_API_KEY',
            'REDDIT_CLIENT_ID',
            'REDDIT_CLIENT_SECRET',
            'REDDIT_USER_AGENT'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            logger.info("Please set the following environment variables:")
            logger.info("  YOUTUBE_API_KEY=your_youtube_api_key")
            logger.info("  REDDIT_CLIENT_ID=your_reddit_client_id")
            logger.info("  REDDIT_CLIENT_SECRET=your_reddit_secret")
            logger.info("  REDDIT_USER_AGENT=your_app_name")
            sys.exit(1)
    
    def _initialize_clients(self):
        """Initialize API clients with credentials."""
        try:
            # YouTube API
            self.youtube_api_key = os.getenv('YOUTUBE_API_KEY')
            self.youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
            logger.info("✓ YouTube API client initialized")
            
            # Reddit API (PRAW)
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT'),
                requestor_kwargs={"timeout": 60}
            )
            logger.info("✓ Reddit API client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize API clients: {e}")
            sys.exit(1)
    
    def scrape_youtube_comments(
        self, 
        video_id: str, 
        max_results: int = 100
    ) -> pd.DataFrame:
        """
        Scrape comments from a YouTube video.
        
        Args:
            video_id: YouTube video ID
            max_results: Maximum number of comments to retrieve per page
            
        Returns:
            DataFrame with columns: comment, author, likes, published_at
        """
        logger.info(f"Scraping YouTube video: {video_id}")
        comments_data = []
        
        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_results,
                textFormat="plainText"
            )
            
            while request:
                response = request.execute()
                
                for item in response['items']:
                    comment_data = item['snippet']['topLevelComment']['snippet']
                    comments_data.append({
                        'comment': comment_data['textDisplay'],
                        'author': comment_data['authorDisplayName'],
                        'likes': comment_data['likeCount'],
                        'published_at': comment_data['publishedAt']
                    })
                
                # Check for next page
                if 'nextPageToken' in response:
                    request = self.youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        pageToken=response['nextPageToken'],
                        maxResults=max_results,
                        textFormat="plainText"
                    )
                else:
                    break
            
            df = pd.DataFrame(comments_data)
            logger.info(f"✓ Scraped {len(df)} YouTube comments")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping YouTube: {e}")
            return pd.DataFrame()
    
    def scrape_reddit_thread(
        self, 
        thread_url: str,
        include_replies: bool = True
    ) -> pd.DataFrame:
        """
        Scrape comments from a Reddit thread.
        
        Args:
            thread_url: Full URL to Reddit thread
            include_replies: Whether to include nested replies
            
        Returns:
            DataFrame with columns: comment, author, score, created_utc
        """
        logger.info(f"Scraping Reddit thread: {thread_url}")
        comments_data = []
        
        try:
            submission = self.reddit.submission(url=thread_url)
            
            if include_replies:
                submission.comments.replace_more(limit=None)
                comments = submission.comments.list()
            else:
                comments = submission.comments
            
            for comment in comments:
                if hasattr(comment, 'body'):
                    comments_data.append({
                        'comment': comment.body,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'score': comment.score,
                        'created_utc': datetime.fromtimestamp(comment.created_utc)
                    })
            
            df = pd.DataFrame(comments_data)
            logger.info(f"✓ Scraped {len(df)} Reddit comments")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping Reddit: {e}")
            return pd.DataFrame()
    
    def scrape_multiple_reddit_threads(
        self, 
        thread_urls: List[str]
    ) -> pd.DataFrame:
        """
        Scrape comments from multiple Reddit threads.
        
        Args:
            thread_urls: List of Reddit thread URLs
            
        Returns:
            Combined DataFrame from all threads
        """
        all_comments = []
        
        for url in thread_urls:
            df = self.scrape_reddit_thread(url)
            if not df.empty:
                df['source_url'] = url
                all_comments.append(df)
        
        if all_comments:
            combined_df = pd.concat(all_comments, ignore_index=True)
            logger.info(f"✓ Total comments from {len(thread_urls)} threads: {len(combined_df)}")
            return combined_df
        
        return pd.DataFrame()
    
    def save_to_csv(
        self, 
        df: pd.DataFrame, 
        filename: str,
        output_dir: str = 'data'
    ) -> str:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Output filename (without extension)
            output_dir: Directory to save file in
            
        Returns:
            Full path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(output_dir, f"{filename}_{timestamp}.csv")
        
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"✓ Saved {len(df)} records to {filepath}")
        
        return filepath


def main():
    """Example usage of the SentimentScraper class."""
    
    # Initialize scraper
    scraper = SentimentScraper()
    
    # Example 1: Scrape YouTube comments
    youtube_video_id = "uUo5gnaYB_w"  # Replace with actual video ID
    yt_df = scraper.scrape_youtube_comments(youtube_video_id, max_results=100)
    
    if not yt_df.empty:
        scraper.save_to_csv(yt_df, "youtube_comments")
    
    # Example 2: Scrape Reddit threads
    reddit_urls = [
        "https://www.reddit.com/r/xbox/comments/1l6iqw5/call_of_duty_black_ops_7_official_teaser/",
        "https://www.reddit.com/r/Games/comments/1l6iqtk/call_of_duty_black_ops_7_official_teaser/"
    ]
    
    reddit_df = scraper.scrape_multiple_reddit_threads(reddit_urls)
    
    if not reddit_df.empty:
        scraper.save_to_csv(reddit_df, "reddit_comments")
    
    # Display summary statistics
    print("\n" + "="*50)
    print("SCRAPING SUMMARY")
    print("="*50)
    print(f"YouTube Comments: {len(yt_df)}")
    print(f"Reddit Comments: {len(reddit_df)}")
    print(f"Total Data Points: {len(yt_df) + len(reddit_df)}")
    print("="*50)


if __name__ == "__main__":
    main()
