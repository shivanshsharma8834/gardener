"""
Formula 1 MCP Server for Puch.ai
Comprehensive F1 data access through Jolpica F1 API (Ergast compatible)
Supports all API routes: seasons, circuits, races, constructors, drivers, results, sprint, qualifying, pitstops, laps, standings, and status
"""

import asyncio
import os
import json
import random
import logging
from typing import Annotated, Optional
from datetime import datetime, timezone
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('F1_MCP_Server')

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field

# Data processing imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("WARNING: Pandas not available. Limited data processing.")

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
MCP_SERVER_HOST = os.environ.get("MCP_SERVER_HOST", "0.0.0.0")
MCP_SERVER_PORT = int(os.environ.get("MCP_SERVER_PORT", "8087"))
JOLPICA_BASE_URL = os.environ.get("JOLPICA_BASE_URL", "https://api.jolpi.ca/ergast/f1")

# Current F1 season (hardcoded as requested)
CURRENT_YEAR = 2025

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider (copied from bookmark-reminder-bot) ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Tool descriptions now use simple strings for better MCP compatibility ---

# --- MCP Server Setup ---
mcp = FastMCP(
    "Formula 1 MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Helper Functions ---
async def make_jolpica_request(endpoint: str) -> dict:
    """Make request to Jolpica F1 API with enhanced error handling and logging"""
    logger.info(f"Making API request to endpoint: {endpoint}")
    try:
        url = f"{JOLPICA_BASE_URL}/{endpoint.lstrip('/')}.json"
        logger.info(f"Full URL: {url}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info("Sending HTTP GET request...")
            response = await client.get(url)
            logger.info(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            
            json_data = response.json()
            logger.info(f"Response received - Data structure keys: {list(json_data.keys()) if isinstance(json_data, dict) else 'Not a dict'}")
            
            # Log the structure of MRData if it exists
            if isinstance(json_data, dict) and 'MRData' in json_data:
                mr_data = json_data['MRData']
                logger.info(f"MRData keys: {list(mr_data.keys()) if isinstance(mr_data, dict) else 'MRData not a dict'}")
                
                # Log specific table information
                for table_key in ['DriverTable', 'RaceTable', 'StandingsTable']:
                    if table_key in mr_data:
                        table_data = mr_data[table_key]
                        if isinstance(table_data, dict):
                            logger.info(f"{table_key} keys: {list(table_data.keys())}")
                            # Log count of items
                            for item_key in ['Drivers', 'Races', 'StandingsLists']:
                                if item_key in table_data and isinstance(table_data[item_key], list):
                                    logger.info(f"{table_key}.{item_key} count: {len(table_data[item_key])}")
            else:
                logger.warning("API response does not contain expected 'MRData' key")
                logger.debug(f"Raw response preview: {str(json_data)[:500]}...")
            
            return json_data
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code} for endpoint {endpoint}: {e.response.text}")
        raise McpError(ErrorData(
            code=INTERNAL_ERROR,
            message=f"Jolpica F1 API HTTP error {e.response.status_code}: {e.response.text}"
        ))
    except httpx.RequestError as e:
        logger.error(f"Network error for endpoint {endpoint}: {str(e)}")
        raise McpError(ErrorData(
            code=INTERNAL_ERROR,
            message=f"Jolpica F1 API network error: {str(e)}"
        ))
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for endpoint {endpoint}: {str(e)}")
        raise McpError(ErrorData(
            code=INTERNAL_ERROR,
            message=f"Invalid JSON response from Jolpica F1 API: {str(e)}"
        ))
    except Exception as e:
        logger.error(f"Unexpected error for endpoint {endpoint}: {str(e)}", exc_info=True)
        raise McpError(ErrorData(
            code=INTERNAL_ERROR,
            message=f"Jolpica F1 API unexpected error: {str(e)}"
        ))

def format_race_datetime(date_str: str, time_str: Optional[str] = None) -> str:
    """Format race date and time for display"""
    logger.debug(f"Formatting race datetime - date: {date_str}, time: {time_str}")
    try:
        if date_str:
            if time_str:
                # Handle different time formats from API
                if time_str.endswith('Z'):
                    # Format: "05:10:00Z"
                    datetime_str = f"{date_str}T{time_str}"
                elif '+' in time_str or '-' in time_str:
                    # Format: "05:10:00+00:00" 
                    datetime_str = f"{date_str}T{time_str}"
                else:
                    # Format: "05:10:00" (assume UTC)
                    datetime_str = f"{date_str}T{time_str}+00:00"
                
                dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                formatted = dt.strftime("%B %d, %Y at %H:%M UTC")
                logger.debug(f"Formatted datetime with time: {formatted}")
                return formatted
            else:
                dt = datetime.fromisoformat(date_str)
                formatted = dt.strftime("%B %d, %Y")
                logger.debug(f"Formatted date only: {formatted}")
                return formatted
        logger.warning("Empty date string provided, returning TBD")
        return "TBD"
    except Exception as e:
        logger.error(f"Error formatting race datetime: {str(e)}")
        return f"{date_str} {time_str}" if time_str else date_str

async def get_driver_career_standings(driver_id: str) -> dict:
    """Get driver career standings from recent seasons (API workaround)"""
    logger.info(f"Fetching career standings for driver: {driver_id}")
    
    try:
        current_year = datetime.now().year
        recent_standings = []
        
        # Try to get standings for recent years (last 10 years)
        for year in range(current_year, current_year - 10, -1):
            try:
                year_data = await make_jolpica_request(f"{year}/drivers/{driver_id}/driverStandings")
                if year_data and 'MRData' in year_data:
                    standings_table = year_data['MRData'].get('StandingsTable', {})
                    standings_lists = standings_table.get('StandingsLists', [])
                    if standings_lists:
                        recent_standings.extend(standings_lists)
                        logger.debug(f"Found standings for {driver_id} in {year}")
            except Exception as e:
                logger.debug(f"No standings data for {driver_id} in {year}: {str(e)}")
                continue
        
        # Create mock response structure if we found data
        if recent_standings:
            logger.info(f"Found {len(recent_standings)} seasons of standings data for {driver_id}")
            return {
                'MRData': {
                    'StandingsTable': {
                        'StandingsLists': recent_standings
                    }
                }
            }
        else:
            logger.warning(f"No standings data found for {driver_id}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to fetch career standings for {driver_id}: {str(e)}")
        return None

# --- Load F1 Trivia Data ---
F1_TRIVIA = [
    "Lewis Hamilton holds the record for most pole positions with 104!",
    "Michael Schumacher won 7 World Championships (1994-1995, 2000-2004)",
    "The fastest F1 lap ever was 1:14.260 by Lewis Hamilton at Silverstone 2020",
    "Ferrari is the oldest team in F1, competing since 1950",
    "The most expensive F1 car ever was the McLaren MP4/1 at $50 million",
    "The shortest F1 race was the 2021 Belgian GP - just 3 laps behind safety car",
    "F1 cars can accelerate from 0-200 km/h in less than 5 seconds",
    "Monaco GP is the most prestigious race, held since 1929",
    "F1 engines reach temperatures of over 1000°C during races",
    "Sebastian Vettel won 4 consecutive championships (2010-2013) with Red Bull",
    "The longest F1 race was the 2011 Canadian GP at 4 hours and 4 minutes",
    "Ayrton Senna is considered one of the greatest drivers, with 41 wins and 3 championships",
    "DRS (Drag Reduction System) was introduced in 2011 to increase overtaking",
    "The 2020 Turkish GP saw the first intermediate tire win since 2008",
    "Max Verstappen became the youngest F1 winner at 18 years and 228 days"
]

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    """Validate the MCP server connection and return user number"""
    logger.info("Validating MCP server connection")
    if MY_NUMBER is None:
        logger.error("MY_NUMBER environment variable is not set")
        raise ValueError("MY_NUMBER environment variable must be set")
    logger.info(f"Validation successful, returning number: {MY_NUMBER}")
    return MY_NUMBER

# --- F1 MCP Tools ---

# Tool 1: Get Next Race

@mcp.tool(description="Get details about the next upcoming Formula 1 race including date, time, circuit, and location. Use when user asks about the next F1 race or upcoming races.")
async def get_next_race() -> str:
    """Get the next upcoming F1 race details"""
    logger.info("Fetching next race details")
    try:
        # Get current date
        current_date = datetime.now(timezone.utc).isoformat()[:10]
        logger.debug(f"Current date: {current_date}")
        
        # Get current race schedule
        logger.debug("Fetching current race schedule")
        data = await make_jolpica_request("current")
        
        if not data or 'MRData' not in data or 'RaceTable' not in data['MRData']:
            logger.warning("No race data found in API response")
            return "🏁 No race data found. API might be unavailable."
        
        races = data['MRData']['RaceTable']['Races']
        if not races:
            logger.warning("Empty race list in API response")
            return "🏁 No upcoming races found. Season might be over."
        
        # Find next race (first one in the current season)
        current_date = datetime.now(timezone.utc).date()
        next_race = None
        
        logger.debug("Searching for next race")
        for race in races:
            race_date = datetime.fromisoformat(race['date']).date()
            logger.debug(f"Checking race: {race['raceName']} on {race_date}")
            if race_date >= current_date:
                next_race = race
                logger.info(f"Found next race: {race['raceName']} on {race_date}")
                break
        
        if not next_race:
            logger.warning("No upcoming races found")
            return "🏁 No upcoming races found for this season."
        
        # Format response
        circuit = next_race['Circuit']
        location = circuit['Location']
        
        response = f"""🏁 **Next F1 Race**

**{next_race['raceName']}**
**Location**: {location['locality']}, {location['country']}
**Circuit**: {circuit['circuitName']}
**Date**: {format_race_datetime(next_race['date'], next_race.get('time'))}
**🏁 Round**: {next_race['round']}

Get ready for some wheel-to-wheel action! 🏎️💨"""

        return response
        
    except Exception as e:
        return f"Failed to get next race: {str(e)}"

# Tool 2: Get Current Driver Standings

@mcp.tool(description="Get Formula 1 driver championship standings for current season or any specific year. Use when user asks for driver standings, championship points, or season rankings. Use year parameter for historical data (e.g., 2023, 2022, 2021).")
async def get_current_standings(
    year: Annotated[Optional[int], Field(description="Season year (e.g., 2023, 2022, 2021) or leave empty for current season")] = None,
) -> str:
    """Get F1 driver championship standings for current season or specific year"""
    try:
        # Determine API endpoint based on year parameter
        if year is None:
            endpoint = "current/driverStandings"
        else:
            endpoint = f"{year}/driverStandings"
        
        data = await make_jolpica_request(endpoint)
        
        if not data or 'MRData' not in data or 'StandingsTable' not in data['MRData']:
            return f"No driver standings available for {year if year else 'current season'}. API might be unavailable."
        
        standings_list = data['MRData']['StandingsTable']['StandingsLists']
        if not standings_list:
            return f"No driver standings found for {year if year else 'current season'}."
        
        standings = standings_list[0]['DriverStandings']
        season = data['MRData']['StandingsTable']['season']
        
        # Format response
        response = f"**F1 {season} Driver Championship Standings**\n\n"
        
        for standing in standings:
            driver = standing['Driver']
            constructor = standing['Constructors'][0] if standing['Constructors'] else {'name': 'Unknown'}
            
            response += f"**P{standing['position']}: {driver['givenName']} {driver['familyName']}**\n"
            response += f"Team: {constructor['name']}\n"
            response += f"Points: {standing['points']} | Wins: {standing['wins']}\n"
            response += f"Nationality: {driver['nationality']}\n\n"
        
        return response
        
    except Exception as e:
        return f"Failed to get standings: {str(e)}"

# Tool 3: Get Race Schedule

@mcp.tool(description="Get the complete Formula 1 race schedule/calendar for the CURRENT season (2025). Use ONLY when user asks about current/upcoming races without specifying a year.")
async def get_race_schedule() -> str:
    """Get F1 race schedule for current season"""
    try:
        current_year = datetime.now().year
        
        # Get current race schedule
        data = await make_jolpica_request("current")
        
        if not data or 'MRData' not in data or 'RaceTable' not in data['MRData']:
            return "No race schedule found. API might be unavailable."
        
        races = data['MRData']['RaceTable']['Races']
        season = data['MRData']['RaceTable']['season']
        
        if not races:
            return f"No races found for {season} season."
        
        response = f"**F1 {season} Race Calendar**\n\n"
        
        current_date = datetime.now(timezone.utc).date()
        
        for race in races:
            race_date = datetime.fromisoformat(race['date']).date()
            status = "✅" if race_date < current_date else "🔜"
            
            circuit = race['Circuit']
            location = circuit['Location']
            
            response += f"{status} **Round {race['round']}: {race['raceName']}**\n"
            response += f"Location: {location['locality']}, {location['country']}\n"
            response += f"Circuit: {circuit['circuitName']}\n"
            response += f"Date: {format_race_datetime(race['date'], race.get('time'))}\n\n"
        
        return response
        
    except Exception as e:
        return f"Failed to get race schedule: {str(e)}"

# Tool 4: Get Latest Race Results

@mcp.tool(description="Get the latest Formula 1 race results and positions. Use when user asks about race results, who won the last race, or latest race positions.")
async def get_latest_race_results() -> str:
    """Get latest F1 race results"""
    try:
        # Get latest race results
        data = await make_jolpica_request("current/last/results")
        
        if not data or 'MRData' not in data or 'RaceTable' not in data['MRData']:
            return "🏁 No recent race results available. API might be unavailable."
        
        race_table = data['MRData']['RaceTable']
        if not race_table['Races']:
            return "🏁 No race results found."
        
        race = race_table['Races'][0]
        results = race.get('Results', [])
        
        if not results:
            return "🏁 Race results not available yet."
        
        circuit = race['Circuit']
        location = circuit['Location']
        
        response = f"🏁 **Latest Race Results**\n"
        response += f"**{race['raceName']}** (Round {race['round']})\n"
        response += f"Location: {location['locality']}, {location['country']}\n"
        response += f"Circuit: {circuit['circuitName']}\n"
        response += f"Date: {format_race_datetime(race['date'], race.get('time'))}\n\n"
        
        response += "**Final Positions:**\n"
        for result in results:
            driver = result['Driver']
            constructor = result['Constructor']
            
            response += f"**P{result['position']}: {driver['givenName']} {driver['familyName']}** ({constructor['name']})\n"
            
            # Add time/status
            if 'Time' in result:
                response += f"   Time: {result['Time']['time']}\n"
            elif 'status' in result:
                response += f"   Status: {result['status']}\n"
            
            # Add points
            if 'points' in result:
                response += f"   Points: {result['points']}\n"
            
            response += "\n"
        
        return response
        
    except Exception as e:
        return f"Failed to get race results: {str(e)}"

# Tool 5: F1 Trivia

@mcp.tool(description="Get random Formula 1 trivia facts, statistics, and interesting information. Use when user asks for F1 trivia, facts, or wants to learn something interesting about F1.")
async def f1_trivia() -> str:
    """Get random F1 trivia and facts"""
    try:
        trivia = random.choice(F1_TRIVIA)
        
        response = f"""🧠 **F1 Trivia Time!**

{trivia}

💡 Want more F1 facts? Just ask for another trivia!"""
        
        return response
        
    except Exception as e:
        return f"Failed to get F1 trivia: {str(e)}"

# --- Comprehensive Jolpica F1 API Tools ---

# ========== DRIVER ANALYSIS TOOLS ==========

# Tool 6: Get Historical Season Schedule

@mcp.tool(description="Get Formula One race calendar for ANY SPECIFIC YEAR (e.g., 2024, 2023, 2022, 2021, etc.). Use when user mentions a specific year like 'F1 2022 schedule' or 'F1 2021 calendar'. Always use this for historical years.")
async def get_historical_schedule(
    year: Annotated[int, Field(description="Season year (e.g., 2023, 2022, 2021)")] = 2024,
) -> str:
    """Get Formula One race calendar for a specific season"""
    try:
        # Get race schedule for specified year
        data = await make_jolpica_request(f"{year}")
        
        if not data or 'MRData' not in data or 'RaceTable' not in data['MRData']:
            return f"No race schedule found for {year}. API might be unavailable."
        
        races = data['MRData']['RaceTable']['Races']
        season = data['MRData']['RaceTable']['season']
        
        if not races:
            return f"No races found for {season} season."
        
        response = f"**F1 {season} Calendar** ({len(races)} races)\n\n"
        
        # Ultra-compact format to avoid MCP session timeout
        for race in races:
            location = race['Circuit']['Location']
            # Just show date without time for brevity
            date_only = race['date']
            
            response += f"R{race['round']}: {race['raceName']} - {location['locality']} ({date_only})\n"
        
        return response
        
    except Exception as e:
        return f"Failed to get {year} schedule: {str(e)}"

# Tool 7: Get Constructor Standings

@mcp.tool(description="Get Formula One constructor (team) championship standings for current season or any specific year. Use when user asks for constructor standings, team championship, or which team is leading. Use year parameter for historical data (e.g., 2023, 2022, 2021).")
async def get_constructor_standings(
    year: Annotated[Optional[int], Field(description="Season year (e.g., 2023, 2022, 2021) or leave empty for current season")] = None,
) -> str:
    """Get F1 constructor championship standings for current season or specific year"""
    try:
        # Determine API endpoint based on year parameter
        if year is None:
            endpoint = "current/constructorStandings"
        else:
            endpoint = f"{year}/constructorStandings"
        
        data = await make_jolpica_request(endpoint)
        
        if not data or 'MRData' not in data or 'StandingsTable' not in data['MRData']:
            return f"No constructor standings available for {year if year else 'current season'}. API might be unavailable."
        
        standings_list = data['MRData']['StandingsTable']['StandingsLists']
        if not standings_list:
            return f"No constructor standings found for {year if year else 'current season'}."
        
        standings = standings_list[0]['ConstructorStandings']
        season = data['MRData']['StandingsTable']['season']
        
        # Format response
        response = f"**F1 {season} Constructor Championship Standings**\n\n"
        
        for standing in standings:
            constructor = standing['Constructor']
            
            response += f"**P{standing['position']}: {constructor['name']}**\n"
            response += f"Nationality: {constructor['nationality']}\n"
            response += f"Points: {standing['points']} | Wins: {standing['wins']}\n\n"
        
        return response
        
    except Exception as e:
        return f"Failed to get constructor standings: {str(e)}"

# ========== DRIVER ANALYSIS TOOLS ==========

# Tool 8: Get Driver Profile & Career Stats

@mcp.tool(description="Get comprehensive Formula 1 driver profile with career statistics, championship history, race wins, and biographical information. Use when user asks about a specific F1 driver's profile, career stats, or detailed driver information.")
async def get_driver_profile(
    driver_id: Annotated[str, Field(description="REQUIRED: Driver ID or surname. Use lowercase surnames like 'hamilton', 'verstappen', 'leclerc', 'russell', 'rosberg', 'schumacher', etc.")] = "hamilton",
) -> str:
    """Get comprehensive F1 driver profile and career statistics"""
    logger.info(f"Fetching driver profile for: {driver_id}")
    
    try:
        # Get driver basic info
        driver_data = await make_jolpica_request(f"drivers/{driver_id}")
        
        # Validate driver data structure
        if not driver_data or 'MRData' not in driver_data:
            logger.error(f"Invalid API response structure for driver: {driver_id}")
            return f"❌ Invalid API response for driver '{driver_id}'"
        
        driver_table = driver_data['MRData'].get('DriverTable', {})
        drivers = driver_table.get('Drivers', [])
        
        if not drivers:
            logger.warning(f"No driver found with ID: {driver_id}")
            return f"❌ No driver found with ID '{driver_id}'. Try using driver surname like 'hamilton', 'verstappen', 'leclerc'."
        
        driver = drivers[0]
        driver_name = f"{driver.get('givenName', 'Unknown')} {driver.get('familyName', 'Driver')}"
        logger.info(f"Found driver: {driver_name}")
        
        try:
            # Get driver's championship standings (career) - using helper function
            logger.info(f"Fetching championship standings for {driver_name}")
            standings_data = await get_driver_career_standings(driver_id)
            
            # Get driver's race results (wins and career summary)
            logger.info(f"Fetching race results for {driver_name}")
            results_data = await make_jolpica_request(f"drivers/{driver_id}/results")
        except Exception as data_fetch_error:
            logger.warning(f"Failed to fetch some driver data: {str(data_fetch_error)}")
            standings_data = None
            results_data = None
        
        response = f"**{driver_name} - F1 Career Profile**\n\n"
        
        # Basic Info
        response += f"**📊 Personal Information:**\n"
        if 'permanentNumber' in driver:
            response += f"Car Number: #{driver['permanentNumber']}\n"
        if 'code' in driver:
            response += f"Driver Code: {driver['code']}\n"
        response += f"Nationality: {driver['nationality']}\n"
        if 'dateOfBirth' in driver:
            response += f"Date of Birth: {driver['dateOfBirth']}\n"
        response += "\n"
        
        # Championship History with better error handling
        championships = 0
        recent_seasons = []
        
        if standings_data and 'MRData' in standings_data:
            standings_table = standings_data['MRData'].get('StandingsTable', {})
            standings_lists = standings_table.get('StandingsLists', [])
            
            if standings_lists:
                logger.info(f"Processing {len(standings_lists)} seasons of standings data")
                response += f"**🏆 Championship History:**\n"
                
                # Process all seasons but show only last 5
                for season_standings in standings_lists:
                    driver_standings = season_standings.get('DriverStandings', [])
                    if driver_standings:
                        standing = driver_standings[0]
                        season = season_standings.get('season', 'Unknown')
                        
                        try:
                            position = standing.get('position', 'N/A')
                            points = int(standing.get('points', 0))
                            wins = int(standing.get('wins', 0))
                            
                            if position == '1':
                                championships += 1
                            
                            recent_seasons.append((season, position, points, wins))
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error processing season {season} data: {str(e)}")
                            continue
                
                if championships > 0:
                    response += f"🏆 World Championships: {championships}\n"
                
                if recent_seasons:
                    response += f"Recent Seasons Performance:\n"
                    # Show last 5 seasons
                    for season, pos, pts, wins in recent_seasons[-5:]:
                        response += f"  {season}: P{pos} ({pts} pts, {wins} wins)\n"
                    response += "\n"
            else:
                logger.info(f"No championship standings data available for {driver_name}")
                response += f"**🏆 Championship History:** No data available\n\n"
        else:
            logger.warning(f"Invalid or missing standings data for {driver_name}")
            response += f"**🏆 Championship History:** Data unavailable\n\n"
        
        # Race Results Summary with enhanced error handling
        if results_data and 'MRData' in results_data:
            race_table = results_data['MRData'].get('RaceTable', {})
            races = race_table.get('Races', [])
            
            if races:
                logger.info(f"Processing {len(races)} race results for {driver_name}")
                total_races = len(races)
                wins = 0
                podiums = 0
                points_total = 0
                dnfs = 0
                
                # Count statistics with error handling
                for race in races:
                    for result in race.get('Results', []):
                        position = result.get('position', 'N/A')
                        
                        if position != 'N/A':
                            try:
                                pos_int = int(position)
                                if pos_int == 1:
                                    wins += 1
                                if pos_int <= 3:
                                    podiums += 1
                            except ValueError:
                                logger.warning(f"Invalid position in race result: {position}")
                        else:
                            dnfs += 1
                        
                        # Sum points
                        try:
                            points = int(result.get('points', 0))
                            points_total += points
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid points value: {result.get('points')}")
                
                response += f"**🏁 Career Statistics:**\n"
                response += f"Total Races: {total_races}\n"
                response += f"Race Wins: {wins}\n"
                response += f"Podiums: {podiums}\n"
                response += f"Total Points: {points_total}\n"
                response += f"DNFs: {dnfs}\n"
                
                if total_races > 0:
                    win_rate = (wins / total_races) * 100
                    podium_rate = (podiums / total_races) * 100
                    points_per_race = points_total / total_races
                    response += f"Win Rate: {win_rate:.1f}%\n"
                    response += f"Podium Rate: {podium_rate:.1f}%\n"
                    response += f"Avg Points/Race: {points_per_race:.1f}\n"
                response += "\n"
                
                # Recent wins with error handling
                recent_wins = []
                for race in races[-20:]:  # Check last 20 races for wins
                    for result in race.get('Results', []):
                        if result.get('position') == '1':
                            season = race.get('season', 'Unknown')
                            race_name = race.get('raceName', 'Unknown Race')
                            recent_wins.append((season, race_name))
                
                if recent_wins:
                    response += f"**🏆 Recent Race Wins:**\n"
                    for season, race_name in recent_wins[-5:]:  # Last 5 wins
                        response += f"  {season}: {race_name}\n"
            else:
                logger.info(f"No race results available for {driver_name}")
                response += f"**🏁 Career Statistics:** No race data available\n\n"
        else:
            logger.warning(f"Invalid or missing race results data for {driver_name}")
            response += f"**🏁 Career Statistics:** Data unavailable\n\n"
        
        if 'url' in driver:
            response += f"\n🔗 More info: {driver['url']}"
        
        return response
        
    except Exception as e:
        return f"Failed to get driver profile: {str(e)}"

# Tool 9: Get Driver Season Performance

@mcp.tool(description="Get detailed Formula 1 driver performance for a specific season including all race results, points progression, and season statistics. Use when user asks about a driver's performance in a particular season.")
async def get_driver_season_performance(
    driver_id: Annotated[str, Field(description="Driver ID or surname (e.g., 'hamilton', 'verstappen', 'leclerc', 'norris')")] = "verstappen",
    year: Annotated[int, Field(description="Season year to analyze (e.g., 2024, 2023, 2022)")] = 2024,
) -> str:
    """Get detailed F1 driver performance for a specific season"""
    try:
        # Get driver's results for the season
        results_data = await make_jolpica_request(f"{year}/drivers/{driver_id}/results")
        
        if not results_data or 'MRData' not in results_data:
            return f"❌ No data found for driver '{driver_id}' in {year} season."
        
        race_table = results_data['MRData']['RaceTable']
        races = race_table.get('Races', [])
        
        if not races:
            return f"❌ No race results found for driver '{driver_id}' in {year}."
        
        # Get driver's championship position for that season
        standings_data = await make_jolpica_request(f"{year}/drivers/{driver_id}/driverStandings")
        
        # Get driver basic info
        driver_data = await make_jolpica_request(f"drivers/{driver_id}")
        driver_name = "Unknown Driver"
        if driver_data and 'MRData' in driver_data and driver_data['MRData']['DriverTable']['Drivers']:
            driver = driver_data['MRData']['DriverTable']['Drivers'][0]
            driver_name = f"{driver['givenName']} {driver['familyName']}"
        
        response = f"🏎️ **{driver_name} - {year} Season Performance**\n\n"
        
        # Season Summary
        total_races = len(races)
        wins = 0
        podiums = 0
        points_finishes = 0
        dnfs = 0
        total_points = 0
        best_finish = 99
        
        race_results = []
        
        for race in races:
            if race['Results']:
                result = race['Results'][0]  # Driver's result in this race
                position = result.get('position', 'N/A')
                points = int(result.get('points', 0))
                status = result.get('status', 'Unknown')
                
                if position != 'N/A':
                    pos_int = int(position)
                    if pos_int == 1:
                        wins += 1
                    if pos_int <= 3:
                        podiums += 1
                    if points > 0:
                        points_finishes += 1
                    if pos_int < best_finish:
                        best_finish = pos_int
                else:
                    dnfs += 1
                
                total_points += points
                race_results.append((race['round'], race['raceName'], position, points, status))
        
        # Championship Standing
        final_position = "N/A"
        if standings_data and 'MRData' in standings_data:
            standings_lists = standings_data['MRData']['StandingsTable']['StandingsLists']
            if standings_lists and standings_lists[-1]['DriverStandings']:
                final_position = standings_lists[-1]['DriverStandings'][0]['position']
        
        response += f"**🏆 {year} Championship Performance:**\n"
        response += f"Final Championship Position: P{final_position}\n"
        response += f"Total Points: {total_points}\n"
        response += f"Races Participated: {total_races}\n\n"
        
        response += f"**📊 Season Statistics:**\n"
        response += f"🥇 Wins: {wins}\n"
        response += f"🏆 Podiums: {podiums}\n"
        response += f"📈 Points Finishes: {points_finishes}\n"
        response += f"🚩 DNFs/Retirements: {dnfs}\n"
        response += f"⭐ Best Finish: P{best_finish if best_finish != 99 else 'N/A'}\n"
        
        if total_races > 0:
            response += f"Win Rate: {(wins/total_races)*100:.1f}%\n"
            response += f"Podium Rate: {(podiums/total_races)*100:.1f}%\n"
        response += "\n"
        
        # Race by Race Results (show key races)
        response += f"**🏁 Key Race Results:**\n"
        key_races = []
        
        # Show wins first
        for round_num, race_name, pos, pts, status in race_results:
            if pos == '1':
                key_races.append(f"🏆 R{round_num} {race_name}: P{pos} ({pts} pts)")
        
        # Then podiums
        for round_num, race_name, pos, pts, status in race_results:
            if pos in ['2', '3']:
                key_races.append(f"🥇 R{round_num} {race_name}: P{pos} ({pts} pts)")
        
        # Show first 5 key results
        for result in key_races[:5]:
            response += f"{result}\n"
        
        if len(key_races) > 5:
            response += f"... and {len(key_races) - 5} more strong finishes\n"
        
        return response
        
    except Exception as e:
        return f"Failed to get driver season performance: {str(e)}"

# ========== RACE ANALYSIS TOOLS ==========

# Tool 10: Get Race Analysis (Results + Qualifying)

@mcp.tool(description="Get comprehensive Formula 1 race analysis including qualifying results, race results, fastest laps, and key race statistics. Use when user asks about a specific F1 race analysis or complete race weekend information.")
async def get_race_analysis(
    year: Annotated[int, Field(description="Season year (e.g., 2024, 2023, 2022)")] = 2024,
    round_num: Annotated[int, Field(description="Race round number (1-24) or use 'last' for most recent")] = 1,
) -> str:
    """Get comprehensive F1 race analysis with qualifying and race results"""
    try:
        # Handle 'last' round
        round_identifier = 'last' if round_num == 0 else str(round_num)
        
        # Get race results
        race_data = await make_jolpica_request(f"{year}/{round_identifier}/results")
        
        # Get qualifying results
        quali_data = await make_jolpica_request(f"{year}/{round_identifier}/qualifying")
        
        if not race_data or 'MRData' not in race_data:
            return f"❌ No race data found for {year} round {round_num}."
        
        race_table = race_data['MRData']['RaceTable']
        races = race_table.get('Races', [])
        
        if not races:
            return f"❌ No race found for {year} round {round_num}."
        
        race = races[0]
        circuit = race['Circuit']
        location = circuit['Location']
        
        response = f"🏁 **{race['raceName']} - {year} Race Analysis**\n\n"
        response += f"**Circuit:** {circuit['circuitName']}\n"
        response += f"**Location:** {location['locality']}, {location['country']}\n"
        response += f"**Date:** {race['date']}\n"
        response += f"🏁 **Round:** {race['round']}\n\n"
        
        # Qualifying Results
        if quali_data and 'MRData' in quali_data and quali_data['MRData']['RaceTable']['Races']:
            quali_race = quali_data['MRData']['RaceTable']['Races'][0]
            if 'QualifyingResults' in quali_race:
                response += f"**⏱️ QUALIFYING RESULTS:**\n"
                for i, result in enumerate(quali_race['QualifyingResults'][:10], 1):
                    driver = result['Driver']
                    constructor = result['Constructor']
                    q3_time = result.get('Q3', result.get('Q2', result.get('Q1', 'N/A')))
                    response += f"P{i}: {driver['givenName']} {driver['familyName']} ({constructor['name']}) - {q3_time}\n"
                response += "\n"
        
        # Race Results
        if 'Results' in race:
            response += f"**🏆 RACE RESULTS:**\n"
            for result in race['Results'][:10]:  # Top 10
                driver = result['Driver']
                constructor = result['Constructor']
                position = result['position']
                points = result.get('points', '0')
                
                # Race time or status
                time_status = ""
                if 'Time' in result:
                    time_status = f"({result['Time']['time']})"
                elif 'status' in result:
                    time_status = f"({result['status']})"
                
                response += f"P{position}: {driver['givenName']} {driver['familyName']} ({constructor['name']}) - {points} pts {time_status}\n"
            
            response += "\n"
            
            # Race Winner Details
            if race['Results']:
                winner = race['Results'][0]
                winner_driver = winner['Driver']
                winner_constructor = winner['Constructor']
                response += f"**🏆 Race Winner:** {winner_driver['givenName']} {winner_driver['familyName']} ({winner_constructor['name']})\n"
                
                if 'FastestLap' in winner:
                    fastest = winner['FastestLap']
                    response += f"**⚡ Fastest Lap:** {fastest.get('Time', {}).get('time', 'N/A')} (Lap {fastest.get('lap', 'N/A')})\n"
        
        return response
        
    except Exception as e:
        return f"Failed to get race analysis: {str(e)}"

# Tool 11: Get Qualifying Results

@mcp.tool(description="Get detailed Formula 1 qualifying session results including Q1, Q2, and Q3 times, grid positions, and qualifying performance analysis. Use when user asks about F1 qualifying results, grid positions, or pole position.")
async def get_qualifying_results(
    year: Annotated[int, Field(description="Season year (e.g., 2024, 2023, 2022)")] = 2024,
    round_num: Annotated[int, Field(description="Race round number (1-24) or 0 for last race")] = 1,
) -> str:
    """Get detailed F1 qualifying results with Q1, Q2, Q3 times"""
    try:
        round_identifier = 'last' if round_num == 0 else str(round_num)
        
        quali_data = await make_jolpica_request(f"{year}/{round_identifier}/qualifying")
        
        if not quali_data or 'MRData' not in quali_data:
            return f"❌ No qualifying data found for {year} round {round_num}."
        
        race_table = quali_data['MRData']['RaceTable']
        races = race_table.get('Races', [])
        
        if not races or 'QualifyingResults' not in races[0]:
            return f"❌ No qualifying results found for {year} round {round_num}."
        
        race = races[0]
        circuit = race['Circuit']
        
        response = f"⏱️ **{race['raceName']} - Qualifying Results**\n\n"
        response += f"**Circuit:** {circuit['circuitName']}\n"
        response += f"📅 **Date:** {race['date']}\n\n"
        
        # Qualifying Results
        response += f"**🏁 STARTING GRID:**\n"
        
        q3_drivers = []
        q2_drivers = []
        q1_drivers = []
        
        for result in race['QualifyingResults']:
            driver = result['Driver']
            constructor = result['Constructor']
            position = result['position']
            
            # Determine which session the driver was eliminated in
            if 'Q3' in result:
                q3_time = result['Q3']
                q3_drivers.append((position, driver, constructor, q3_time, 'Q3'))
            elif 'Q2' in result:
                q2_time = result['Q2']
                q2_drivers.append((position, driver, constructor, q2_time, 'Q2'))
            elif 'Q1' in result:
                q1_time = result['Q1']
                q1_drivers.append((position, driver, constructor, q1_time, 'Q1'))
        
        # Show all results in grid order
        all_results = sorted(q3_drivers + q2_drivers + q1_drivers, key=lambda x: int(x[0]))
        
        for pos, driver, constructor, time, session in all_results:
            session_indicator = "🏆" if session == "Q3" else "⚡" if session == "Q2" else "📊"
            response += f"P{pos}: {session_indicator} {driver['givenName']} {driver['familyName']} ({constructor['name']}) - {time}\n"
        
        response += "\n"
        
        # Pole position highlight
        if q3_drivers:
            pole_sitter = min(q3_drivers, key=lambda x: int(x[0]))
            response += f"**🏆 POLE POSITION:** {pole_sitter[1]['givenName']} {pole_sitter[1]['familyName']} ({pole_sitter[2]['name']}) - {pole_sitter[3]}\n"
        
        response += f"\n🏆 Q3 (Top 10) | ⚡ Q2 (P11-15) | 📊 Q1 (P16-20)"
        
        return response
        
    except Exception as e:
        return f"Failed to get qualifying results: {str(e)}"

# ========== HISTORICAL COMPARISON TOOLS ==========

# Tool 12: Compare Drivers Head-to-Head

@mcp.tool(description="Compare two Formula 1 drivers head-to-head with comprehensive statistics including championship positions, race wins, podiums, and points. Can compare entire careers or specific seasons. Use when user wants to compare F1 drivers or asks who is better between drivers.")
async def compare_drivers(
    driver1_id: Annotated[str, Field(description="REQUIRED: First driver ID or surname. Use lowercase surnames like 'hamilton', 'verstappen', 'leclerc', 'rosberg', etc.")] = "hamilton",
    driver2_id: Annotated[str, Field(description="REQUIRED: Second driver ID or surname. Use lowercase surnames like 'schumacher', 'vettel', 'alonso', 'rosberg', etc.")] = "verstappen",
    year: Annotated[int, Field(description="Season year to compare (e.g., 2013, 2012, 2021) or 0 for career comparison. Use 0 for career comparison.")] = 0,
) -> str:
    """Compare two F1 drivers head-to-head with comprehensive statistics"""
    logger.info(f"Starting driver comparison: {driver1_id} vs {driver2_id}, year: {year if year != 0 else 'career'}")
    
    try:
        # Get both drivers' basic info
        logger.info(f"Fetching driver data for {driver1_id} and {driver2_id}")
        driver1_data = await make_jolpica_request(f"drivers/{driver1_id}")
        driver2_data = await make_jolpica_request(f"drivers/{driver2_id}")
        
        # Validate driver1 data
        if not driver1_data or 'MRData' not in driver1_data:
            logger.error(f"Invalid response structure for driver1: {driver1_id}")
            return f"❌ Invalid API response for driver '{driver1_id}'"
        
        driver_table1 = driver1_data['MRData'].get('DriverTable', {})
        drivers1 = driver_table1.get('Drivers', [])
        
        if not drivers1:
            logger.warning(f"No driver found with ID: {driver1_id}")
            return f"❌ Driver '{driver1_id}' not found. Try using surnames like 'hamilton', 'verstappen', 'leclerc'."
        
        # Validate driver2 data
        if not driver2_data or 'MRData' not in driver2_data:
            logger.error(f"Invalid response structure for driver2: {driver2_id}")
            return f"❌ Invalid API response for driver '{driver2_id}'"
        
        driver_table2 = driver2_data['MRData'].get('DriverTable', {})
        drivers2 = driver_table2.get('Drivers', [])
        
        if not drivers2:
            logger.warning(f"No driver found with ID: {driver2_id}")
            return f"❌ Driver '{driver2_id}' not found. Try using surnames like 'hamilton', 'verstappen', 'leclerc'."
        
        driver1 = drivers1[0]
        driver2 = drivers2[0]
        
        driver1_name = f"{driver1.get('givenName', 'Unknown')} {driver1.get('familyName', 'Driver')}"
        driver2_name = f"{driver2.get('givenName', 'Unknown')} {driver2.get('familyName', 'Driver')}"
        
        logger.info(f"Comparing {driver1_name} vs {driver2_name}")
        
        if year == 0:
            # Career comparison
            response = f"⚔️ **Career Comparison: {driver1_name} vs {driver2_name}**\n\n"
            
            # Get career results for both drivers
            driver1_results = await make_jolpica_request(f"drivers/{driver1_id}/results")
            driver2_results = await make_jolpica_request(f"drivers/{driver2_id}/results")
            
            # Get championship standings for both (career)
            driver1_standings = await get_driver_career_standings(driver1_id)
            driver2_standings = await get_driver_career_standings(driver2_id)
            
        else:
            # Season comparison
            response = f"⚔️ **{year} Season Comparison: {driver1_name} vs {driver2_name}**\n\n"
            
            driver1_results = await make_jolpica_request(f"{year}/drivers/{driver1_id}/results")
            driver2_results = await make_jolpica_request(f"{year}/drivers/{driver2_id}/results")
            
            driver1_standings = await make_jolpica_request(f"{year}/drivers/{driver1_id}/driverStandings")
            driver2_standings = await make_jolpica_request(f"{year}/drivers/{driver2_id}/driverStandings")
        
        # Calculate statistics for both drivers with enhanced error handling
        def calculate_stats(results_data, standings_data, driver_name):
            logger.info(f"Calculating statistics for {driver_name}")
            stats = {
                'races': 0, 'wins': 0, 'podiums': 0, 'points': 0,
                'poles': 0, 'fastest_laps': 0, 'championships': 0,
                'best_championship_pos': 99, 'dnfs': 0
            }
            
            # Process results data
            if results_data and 'MRData' in results_data:
                race_table = results_data['MRData'].get('RaceTable', {})
                races = race_table.get('Races', [])
                stats['races'] = len(races)
                logger.info(f"{driver_name} participated in {len(races)} races")
                
                for race in races:
                    for result in race.get('Results', []):
                        position = result.get('position', 'N/A')
                        
                        # Count valid positions only
                        if position != 'N/A':
                            try:
                                pos_int = int(position)
                                if pos_int == 1:
                                    stats['wins'] += 1
                                if pos_int <= 3:
                                    stats['podiums'] += 1
                            except ValueError:
                                logger.warning(f"Invalid position value: {position}")
                        else:
                            stats['dnfs'] += 1
                        
                        # Add points
                        try:
                            points = int(result.get('points', 0))
                            stats['points'] += points
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid points value: {result.get('points')}")
                        
                        # Count fastest laps
                        if 'FastestLap' in result:
                            fastest_lap = result['FastestLap']
                            if isinstance(fastest_lap, dict) and fastest_lap.get('rank') == '1':
                                stats['fastest_laps'] += 1
            else:
                logger.warning(f"No valid results data for {driver_name}")
            
            # Process standings data
            if standings_data and 'MRData' in standings_data:
                standings_table = standings_data['MRData'].get('StandingsTable', {})
                standings_lists = standings_table.get('StandingsLists', [])
                logger.info(f"{driver_name} has {len(standings_lists)} seasons of standings data")
                
                for season_standings in standings_lists:
                    driver_standings = season_standings.get('DriverStandings', [])
                    if driver_standings:
                        standing = driver_standings[0]
                        try:
                            position = int(standing.get('position', 99))
                            if position == 1:
                                stats['championships'] += 1
                            if position < stats['best_championship_pos']:
                                stats['best_championship_pos'] = position
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid championship position: {standing.get('position')}")
            else:
                logger.warning(f"No valid standings data for {driver_name}")
            
            logger.info(f"{driver_name} stats: {stats}")
            return stats
        
        stats1 = calculate_stats(driver1_results, driver1_standings, driver1_name)
        stats2 = calculate_stats(driver2_results, driver2_standings, driver2_name)
        
        # Format comparison
        response += f"**📊 STATISTICAL COMPARISON:**\n"
        response += f"```\n"
        response += f"{'Metric':<20} {'Driver 1':<15} {'Driver 2':<15} {'Winner'}\n"
        response += f"{'-'*20} {'-'*15} {'-'*15} {'-'*10}\n"
        response += f"{'Races':<20} {stats1['races']:<15} {stats2['races']:<15} {'=' if stats1['races']==stats2['races'] else ('1️⃣' if stats1['races']>stats2['races'] else '2️⃣')}\n"
        response += f"{'Wins':<20} {stats1['wins']:<15} {stats2['wins']:<15} {'=' if stats1['wins']==stats2['wins'] else ('1️⃣' if stats1['wins']>stats2['wins'] else '2️⃣')}\n"
        response += f"{'Podiums':<20} {stats1['podiums']:<15} {stats2['podiums']:<15} {'=' if stats1['podiums']==stats2['podiums'] else ('1️⃣' if stats1['podiums']>stats2['podiums'] else '2️⃣')}\n"
        response += f"{'Points':<20} {stats1['points']:<15} {stats2['points']:<15} {'=' if stats1['points']==stats2['points'] else ('1️⃣' if stats1['points']>stats2['points'] else '2️⃣')}\n"
        response += f"{'Championships':<20} {stats1['championships']:<15} {stats2['championships']:<15} {'=' if stats1['championships']==stats2['championships'] else ('1️⃣' if stats1['championships']>stats2['championships'] else '2️⃣')}\n"
        response += f"```\n\n"
        
        # Win rates
        win_rate1 = (stats1['wins'] / stats1['races'] * 100) if stats1['races'] > 0 else 0
        win_rate2 = (stats2['wins'] / stats2['races'] * 100) if stats2['races'] > 0 else 0
        
        podium_rate1 = (stats1['podiums'] / stats1['races'] * 100) if stats1['races'] > 0 else 0
        podium_rate2 = (stats2['podiums'] / stats2['races'] * 100) if stats2['races'] > 0 else 0
        
        response += f"**🎯 SUCCESS RATES:**\n"
        response += f"{driver1_name}: {win_rate1:.1f}% win rate, {podium_rate1:.1f}% podium rate\n"
        response += f"{driver2_name}: {win_rate2:.1f}% win rate, {podium_rate2:.1f}% podium rate\n"
        
        return response
        
    except Exception as e:
        return f"Failed to compare drivers: {str(e)}"

# ========== NEW JOLPICA API COMPREHENSIVE TOOLS ==========

# Tool 13: Get All Seasons
@mcp.tool(description="Get all Formula 1 seasons/years available in the database. Use when user asks about F1 history, available seasons, or when to browse all F1 years.")
async def get_all_seasons() -> str:
    """Get all available F1 seasons"""
    try:
        data = await make_jolpica_request("seasons")
        
        if not data or 'MRData' not in data or 'SeasonTable' not in data['MRData']:
            return "No season data available."
        
        seasons = data['MRData']['SeasonTable'].get('Seasons', [])
        
        if not seasons:
            return "No seasons found."
        
        response = "**Available F1 Seasons**\n\n"
        
        # Group seasons by decades
        decades = {}
        for season in seasons:
            year = int(season['season'])
            decade = (year // 10) * 10
            if decade not in decades:
                decades[decade] = []
            decades[decade].append(year)
        
        for decade in sorted(decades.keys()):
            years = sorted(decades[decade])
            response += f"**{decade}s:** {', '.join(map(str, years))}\n"
        
        response += f"\n**Total Seasons:** {len(seasons)} (from {seasons[0]['season']} to {seasons[-1]['season']})"
        
        return response
        
    except Exception as e:
        return f"Failed to get seasons: {str(e)}"

# Tool 14: Get All Circuits
@mcp.tool(description="Get all Formula 1 circuits/tracks ever used in F1. Use when user asks about F1 circuits, tracks, or racing venues.")
async def get_all_circuits() -> str:
    """Get all F1 circuits"""
    try:
        data = await make_jolpica_request("circuits")
        
        if not data or 'MRData' not in data or 'CircuitTable' not in data['MRData']:
            return "No circuit data available."
        
        circuits = data['MRData']['CircuitTable'].get('Circuits', [])
        
        if not circuits:
            return "No circuits found."
        
        response = f"**Formula 1 Circuits Database**\n\n"
        response += f"**Total Circuits:** {len(circuits)}\n\n"
        
        # Group by country
        countries = {}
        for circuit in circuits:
            location = circuit.get('Location', {})
            country = location.get('country', 'Unknown')
            if country not in countries:
                countries[country] = []
            countries[country].append(circuit)
        
        for country in sorted(countries.keys()):
            response += f"**{country}:**\n"
            for circuit in sorted(countries[country], key=lambda x: x.get('circuitName', '')):
                location = circuit.get('Location', {})
                locality = location.get('locality', 'Unknown')
                response += f"  • {circuit.get('circuitName', 'Unknown')} ({locality})\n"
            response += "\n"
        
        return response
        
    except Exception as e:
        return f"Failed to get circuits: {str(e)}"

# Tool 15: Get All Drivers (Current Season)
@mcp.tool(description="Get all Formula 1 drivers for current season. Use when user asks about current F1 drivers, driver lineup, or who is racing this year.")
async def get_current_drivers() -> str:
    """Get all current F1 drivers"""
    try:
        data = await make_jolpica_request("current/drivers")
        
        if not data or 'MRData' not in data or 'DriverTable' not in data['MRData']:
            return "No driver data available."
        
        drivers = data['MRData']['DriverTable'].get('Drivers', [])
        season = data['MRData']['DriverTable'].get('season', CURRENT_YEAR)
        
        if not drivers:
            return f"No drivers found for {season} season."
        
        response = f"**{season} F1 Driver Lineup**\n\n"
        response += f"**Total Drivers:** {len(drivers)}\n\n"
        
        # Group by nationality
        nationalities = {}
        for driver in drivers:
            nationality = driver.get('nationality', 'Unknown')
            if nationality not in nationalities:
                nationalities[nationality] = []
            nationalities[nationality].append(driver)
        
        for nationality in sorted(nationalities.keys()):
            response += f"**{nationality}:**\n"
            for driver in sorted(nationalities[nationality], key=lambda x: x.get('familyName', '')):
                name = f"{driver.get('givenName', '')} {driver.get('familyName', '')}"
                number = f" (#{driver['permanentNumber']})" if 'permanentNumber' in driver else ""
                code = f" [{driver['code']}]" if 'code' in driver else ""
                response += f"  • {name.strip()}{number}{code}\n"
            response += "\n"
        
        return response
        
    except Exception as e:
        return f"Failed to get current drivers: {str(e)}"

# Tool 16: Get All Constructors (Current Season)  
@mcp.tool(description="Get all Formula 1 constructors/teams for current season. Use when user asks about current F1 teams, constructors, or team lineup.")
async def get_current_constructors() -> str:
    """Get all current F1 constructors"""
    try:
        data = await make_jolpica_request("current/constructors")
        
        if not data or 'MRData' not in data or 'ConstructorTable' not in data['MRData']:
            return "No constructor data available."
        
        constructors = data['MRData']['ConstructorTable'].get('Constructors', [])
        season = data['MRData']['ConstructorTable'].get('season', CURRENT_YEAR)
        
        if not constructors:
            return f"No constructors found for {season} season."
        
        response = f"**{season} F1 Constructor Lineup**\n\n"
        response += f"**Total Teams:** {len(constructors)}\n\n"
        
        for constructor in sorted(constructors, key=lambda x: x.get('name', '')):
            name = constructor.get('name', 'Unknown')
            nationality = constructor.get('nationality', 'Unknown')
            response += f"**{name}** ({nationality})\n"
            if 'url' in constructor:
                response += f"  Link: {constructor['url']}\n"
            response += "\n"
        
        return response
        
    except Exception as e:
        return f"Failed to get current constructors: {str(e)}"

# Tool 17: Get Sprint Results
@mcp.tool(description="Get Formula 1 sprint race results for a specific season and round. Use when user asks about sprint race results, sprint qualifying, or Saturday race results.")
async def get_sprint_results(
    year: Annotated[int, Field(description="Season year (e.g., 2024, 2023, 2022)")] = 2024,
    round_num: Annotated[int, Field(description="Race round number (1-24) or 0 for last race")] = 1,
) -> str:
    """Get F1 sprint race results"""
    try:
        round_identifier = 'last' if round_num == 0 else str(round_num)
        data = await make_jolpica_request(f"{year}/{round_identifier}/sprint")
        
        if not data or 'MRData' not in data or 'RaceTable' not in data['MRData']:
            return f"No sprint data found for {year} round {round_num}."
        
        races = data['MRData']['RaceTable'].get('Races', [])
        
        if not races or 'SprintResults' not in races[0]:
            return f"No sprint results found for {year} round {round_num}."
        
        race = races[0]
        circuit = race['Circuit']
        location = circuit['Location']
        
        response = f"**{race['raceName']} - Sprint Results**\n\n"
        response += f"**Circuit:** {circuit['circuitName']}\n"
        response += f"**Location:** {location['locality']}, {location['country']}\n"
        response += f"**Date:** {race['date']}\n\n"
        
        response += f"**SPRINT RACE RESULTS:**\n"
        for result in race['SprintResults']:
            driver = result['Driver']
            constructor = result['Constructor']
            position = result['position']
            points = result.get('points', '0')
            
            time_status = ""
            if 'Time' in result:
                time_status = f"({result['Time']['time']})"
            elif 'status' in result:
                time_status = f"({result['status']})"
            
            response += f"P{position}: {driver['givenName']} {driver['familyName']} ({constructor['name']}) - {points} pts {time_status}\n"
        
        return response
        
    except Exception as e:
        return f"Failed to get sprint results: {str(e)}"

# Tool 18: Get Pit Stop Data
@mcp.tool(description="Get Formula 1 pit stop data for a specific race. Use when user asks about pit stops, pit stop strategies, or fastest pit stops.")
async def get_pitstops(
    year: Annotated[int, Field(description="Season year (e.g., 2024, 2023, 2022)")] = 2024,
    round_num: Annotated[int, Field(description="Race round number (1-24)")] = 1,
) -> str:
    """Get F1 pit stop data"""
    try:
        data = await make_jolpica_request(f"{year}/{round_num}/pitstops")
        
        if not data or 'MRData' not in data or 'RaceTable' not in data['MRData']:
            return f"No pit stop data found for {year} round {round_num}."
        
        races = data['MRData']['RaceTable'].get('Races', [])
        
        if not races or 'PitStops' not in races[0]:
            return f"No pit stops found for {year} round {round_num}."
        
        race = races[0]
        circuit = race['Circuit']
        pit_stops = race['PitStops']
        
        response = f"**{race['raceName']} - Pit Stop Analysis**\n\n"
        response += f"**Circuit:** {circuit['circuitName']}\n"
        response += f"**Total Pit Stops:** {len(pit_stops)}\n\n"
        
        # Group by driver
        driver_stops = {}
        fastest_stop = None
        
        for stop in pit_stops:
            driver_id = stop['driverId']
            if driver_id not in driver_stops:
                driver_stops[driver_id] = []
            driver_stops[driver_id].append(stop)
            
            # Track fastest stop
            if 'duration' in stop:
                duration = float(stop['duration'])
                if not fastest_stop or duration < float(fastest_stop.get('duration', 999)):
                    fastest_stop = stop
        
        if fastest_stop:
            response += f"**Fastest Pit Stop:** {fastest_stop['duration']}s (Lap {fastest_stop['lap']})\n\n"
        
        response += f"**PIT STOP SUMMARY BY DRIVER:**\n"
        for driver_id in sorted(driver_stops.keys()):
            stops = driver_stops[driver_id]
            avg_time = sum(float(s.get('duration', 0)) for s in stops) / len(stops)
            response += f"**{driver_id.upper()}:** {len(stops)} stops, avg {avg_time:.2f}s\n"
            for stop in stops:
                response += f"  • Lap {stop['lap']}: {stop.get('duration', 'N/A')}s\n"
            response += "\n"
        
        return response
        
    except Exception as e:
        return f"Failed to get pit stop data: {str(e)}"

# Tool 19: Get Lap Times
@mcp.tool(description="Get Formula 1 lap times for a specific race. Use when user asks about lap times, race pace, or lap-by-lap analysis.")
async def get_lap_times(
    year: Annotated[int, Field(description="Season year (e.g., 2024, 2023, 2022)")] = 2024,
    round_num: Annotated[int, Field(description="Race round number (1-24)")] = 1,
) -> str:
    """Get F1 lap times (limited to avoid overwhelming data)"""
    try:
        data = await make_jolpica_request(f"{year}/{round_num}/laps")
        
        if not data or 'MRData' not in data or 'RaceTable' not in data['MRData']:
            return f"No lap time data found for {year} round {round_num}."
        
        races = data['MRData']['RaceTable'].get('Races', [])
        
        if not races or 'Laps' not in races[0]:
            return f"No lap times found for {year} round {round_num}."
        
        race = races[0]
        circuit = race['Circuit']
        laps = race['Laps']
        
        response = f"**{race['raceName']} - Lap Time Analysis**\n\n"
        response += f"**Circuit:** {circuit['circuitName']}\n"
        response += f"**Total Laps:** {len(laps)}\n\n"
        
        # Find fastest lap overall
        fastest_lap = None
        fastest_time = None
        
        for lap in laps:
            for timing in lap.get('Timings', []):
                if 'time' in timing:
                    time_str = timing['time']
                    # Convert lap time to seconds for comparison
                    try:
                        parts = time_str.split(':')
                        if len(parts) == 2:
                            minutes = int(parts[0])
                            seconds = float(parts[1])
                            total_seconds = minutes * 60 + seconds
                            
                            if not fastest_time or total_seconds < fastest_time:
                                fastest_time = total_seconds
                                fastest_lap = {
                                    'lap': lap['number'],
                                    'driver': timing['driverId'],
                                    'time': time_str
                                }
                    except (ValueError, IndexError):
                        continue
        
        if fastest_lap:
            response += f"**Fastest Lap:** {fastest_lap['time']} by {fastest_lap['driver'].upper()} (Lap {fastest_lap['lap']})\n\n"
        
        # Show sample lap times (first 5 laps)
        response += f"**SAMPLE LAP TIMES (First 5 Laps):**\n"
        for lap in laps[:5]:
            lap_num = lap['number']
            response += f"**Lap {lap_num}:**\n"
            
            timings = lap.get('Timings', [])
            # Show top 5 drivers for this lap
            for timing in timings[:5]:
                driver = timing['driverId'].upper()
                time = timing.get('time', 'N/A')
                position = timing.get('position', 'N/A')
                response += f"  P{position} {driver}: {time}\n"
            response += "\n"
        
        if len(laps) > 5:
            response += f"... and {len(laps) - 5} more laps\n"
        
        return response
        
    except Exception as e:
        return f"Failed to get lap times: {str(e)}"

# Tool 20: Get Status Codes
@mcp.tool(description="Get all Formula 1 status codes used in results (e.g., Finished, Retired, DNF reasons). Use when user asks about F1 status codes or DNF classifications.")
async def get_status_codes() -> str:
    """Get all F1 status codes"""
    try:
        data = await make_jolpica_request("status")
        
        if not data or 'MRData' not in data or 'StatusTable' not in data['MRData']:
            return "No status data available."
        
        statuses = data['MRData']['StatusTable'].get('Status', [])
        
        if not statuses:
            return "No status codes found."
        
        response = f"**Formula 1 Status Codes**\n\n"
        response += f"**Total Status Codes:** {len(statuses)}\n\n"
        
        # Group by category
        finished_statuses = []
        retirement_statuses = []
        other_statuses = []
        
        for status in statuses:
            status_text = status.get('status', '').lower()
            if 'finished' in status_text or status_text == '+1 lap' or status_text.startswith('+'):
                finished_statuses.append(status)
            elif any(word in status_text for word in ['engine', 'gearbox', 'transmission', 'accident', 'collision', 'spun', 'retired', 'withdraw']):
                retirement_statuses.append(status)
            else:
                other_statuses.append(status)
        
        if finished_statuses:
            response += f"**RACE COMPLETION:**\n"
            for status in finished_statuses:
                response += f"  • {status.get('status', 'Unknown')}\n"
            response += "\n"
        
        if retirement_statuses:
            response += f"**RETIREMENTS/DNF:**\n"
            for status in retirement_statuses:
                response += f"  • {status.get('status', 'Unknown')}\n"
            response += "\n"
        
        if other_statuses:
            response += f"**OTHER STATUS:**\n"
            for status in other_statuses:
                response += f"  • {status.get('status', 'Unknown')}\n"
            response += "\n"
        
        return response
        
    except Exception as e:
        return f"Failed to get status codes: {str(e)}"

# --- Run MCP Server ---
async def main():
    try:
        logger.info("=== Formula 1 MCP Server Starting ===")
        logger.info(f"Server URL: http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}")
        logger.info(f"Configured phone number: {MY_NUMBER}")
        logger.info(f"Jolpica F1 API endpoint: {JOLPICA_BASE_URL}")
        
        # Log environment configuration
        logger.info("Environment Configuration:")
        logger.info(f"- Pandas Available: {PANDAS_AVAILABLE}")
        logger.info(f"- Current Year: {CURRENT_YEAR}")
        logger.info(f"- Server Host: {MCP_SERVER_HOST}")
        logger.info(f"- Server Port: {MCP_SERVER_PORT}")
        
        # Log feature availability
        logger.info("Available Features:")
        logger.info("- Real-time race data")
        logger.info("- Historical season analysis")
        logger.info("- Driver/Constructor standings")
        logger.info("- Race results and qualifying data")
        logger.info("- Driver comparisons and statistics")
        
        # Start the server
        logger.info("Starting MCP server...")
        await mcp.run_async("streamable-http", host=MCP_SERVER_HOST, port=MCP_SERVER_PORT)
        logger.info("MCP server started successfully")
        
    except Exception as e:
        logger.critical(f"Failed to start MCP server: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())