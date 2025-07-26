from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
from openai import OpenAI
import secrets
from pathlib import Path
import requests
import time
import re
import random

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Secure session key setup
key_path = Path("instance/secret_key.txt")
if not key_path.exists():
    key_path.parent.mkdir(exist_ok=True)
    key_path.write_text(secrets.token_hex(32))
app.secret_key = key_path.read_text()

# Local Ollama connection
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Scene presets
VISUAL_SCENES = {
    "dragon": {
        "prompt": "Dark fantasy dragon battle, glowing scales, digital painting",
        "negative": "cartoon, anime, blurry"
    },
    "forest": {
        "prompt": "Enchanted twilight forest, 8k game environment",
        "negative": "daytime, city, modern"
    },
    "default": {
        "prompt": "Fantasy story scene, vivid colors, unreal engine style",
        "negative": "low quality"
    }
}

# Dragon behavior patterns based on difficulty
DRAGON_STRATEGIES = {
    "easy": {
        "aggression": 0.4,
        "tactics": ["basic_fire", "claw_swipe", "tail_whip"],
        "special_moves": ["roar"],
        "counter_probability": 0.3
    },
    "normal": {
        "aggression": 0.6,
        "tactics": ["fire_breath", "wing_buffet", "bite", "charge"],
        "special_moves": ["flame_burst", "intimidating_roar"],
        "counter_probability": 0.5
    },
    "hard": {
        "aggression": 0.8,
        "tactics": ["inferno_breath", "crushing_bite", "ground_slam", "aerial_dive"],
        "special_moves": ["dragon_rage", "flame_tornado", "ancient_curse"],
        "counter_probability": 0.7
    }
}

def get_enhanced_game_prompt(game_state, player_action):
    """
    Enhanced prompt engineering to ensure consistent stat reporting
    """
    # Convert your game state format to match the enhanced prompt expectations
    player_health = game_state.get('playerHealth', 100)
    player_mana = game_state.get('playerMana', 50)
    dragon_health = game_state.get('dragonHealth', 150)
    current_location = game_state.get('location', 'cave')  # Default to cave since it's a dragon battle
    inventory = game_state.get('inventory', {})
    
    base_prompt = f"""You are the Game Master of an epic fantasy RPG. The player is on a quest to defeat a mighty dragon.

CRITICAL INSTRUCTION - ALWAYS include these EXACT stat change patterns in your response:
- For player damage:"health -[number]"
- For dragon damage:"Dragon health -[number]"
- For mana usage: "You spend [number] mana"
- For mana gain: "You recover [number] mana"
- For health restoration: "Health restored: [number]" or "You heal [number] health"
- For inventory gains: "Find [number] [item]" or "Gain [number] [item]"

GAME STATE:
- Player Health: {player_health}/100
- Player Mana: {player_mana}/100
- Dragon Health: {dragon_health}/200
- Current Location: {current_location}
- Inventory: {inventory}

PLAYER ACTION: {player_action}

RESPONSE GUIDELINES:
1. Always describe combat with specific damage numbers
2. Include exact mana costs for spells/abilities
3. Mention health/mana recovery amounts precisely
4. Use consistent terminology for stat changes
5. Make the story engaging but include required stat patterns

EXAMPLE RESPONSES:
- "Your sword strikes true! Dragon takes 35 damage, roaring in pain as scales shatter."
- "The fireball spell drains your energy. Mana -25. The dragon staggers as flames engulf it. Dragon takes 40 damage."
- "You drink a healing potion. Health restored: 30. You feel renewed strength flow through you."
- "The dragon's claw rakes across your armor. You take 18 damage but stand firm."

Respond with a vivid, immersive narrative that includes the required stat change patterns."""

    # Add location-specific enhancements
    location_prompts = {
        'village': """
LOCATION CONTEXT: You are in a peaceful village. Focus on:
- Healing opportunities: "The village healer tends to you. Health restored: 25"
- Item purchases: "You buy 2 health potions from the merchant"
- Mana restoration: "Rest at the inn restores your energy. Mana +30"
""",
        'forest': """
LOCATION CONTEXT: You are in a dark, mystical forest. Include:
- Environmental hazards: "Thorns scratch you. You take 8 damage"
- Foraging opportunities: "You find 1 mana potion hidden under roots"
- Mysterious encounters that might restore or drain stats
""",
        'cave': """
LOCATION CONTEXT: You are in the dragon's cave. Emphasize:
- Intense combat: "Dragon breathes fire! You take 45 damage"
- Treasure discoveries: "Find 3 gold coins in the dragon's hoard"
- High-stakes magical battles with significant mana costs
""",
        'mountain': """
LOCATION CONTEXT: You are on a treacherous mountain. Include:
- Climbing hazards: "You slip on loose rocks. You take 12 damage"
- Rare herb discoveries: "Find 1 rare healing herb. Health restored: 35"
- Altitude effects on mana recovery
"""
    }
    
    current_location_lower = current_location.lower()
    if current_location_lower in location_prompts:
        base_prompt += location_prompts[current_location_lower]
    
    # Add combat intensity based on dragon health
    if dragon_health < 50:
        base_prompt += """
COMBAT INTENSITY: The dragon is nearly defeated and desperate! Include:
- Higher damage attacks: "Dragon's fury unleashes! You take 60 damage"
- Powerful final abilities that cost significant mana
- Opportunities for finishing moves with specific damage numbers
"""
    elif dragon_health < 100:
        base_prompt += """
COMBAT INTENSITY: The dragon is wounded and angry! Include:
- Moderate to high damage exchanges
- Strategic spell usage with clear mana costs
- Balanced combat with specific damage values
"""
    
    # Add mana-based action suggestions
    if player_mana < 20:
        base_prompt += """
MANA STATUS: Player has low mana. Suggest:
- Mana restoration opportunities: "You meditate briefly. Mana +15"
- Physical attacks that don't cost mana
- Items that restore mana
"""
    elif player_mana > 80:
        base_prompt += """
MANA STATUS: Player has high mana. Encourage:
- Powerful spells with high mana costs: "Cast lightning bolt. Mana -35"
- Magical abilities that deal significant damage
- Elaborate magical effects
"""
    
    # Add health-based urgency
    if player_health < 30:
        base_prompt += """
HEALTH STATUS: Player is critically injured! Include:
- Urgent healing opportunities
- Higher stakes in combat descriptions
- Specific healing amounts when available
"""
    
    return base_prompt

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/start_session", methods=["POST"])
def start_session():
    """Initialize or reset story session"""
    session.clear()
    session['history'] = []
    session['last_image_scene'] = None
    session['turn_count'] = 0
    session['dragon_state'] = "aggressive"  # aggressive, wounded, desperate, dying
    return jsonify({"status": "New session started"})

@app.route("/generate", methods=["POST"])
def generate():
    if 'history' not in session:
        start_session()

    data = request.get_json()
    user_prompt = data.get("prompt", "").strip()
    game_state = data.get("gameState", {})

    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Analyze the player's action and game state
        action_analysis = analyze_player_action(user_prompt, game_state)
        
        # Generate enhanced system prompt using the new integrated system
        enhanced_game_prompt = get_enhanced_game_prompt(game_state, user_prompt)
        
        # Keep your existing dynamic system prompt but enhance it
        system_prompt = create_dynamic_system_prompt(game_state, action_analysis)
        
        # Combine both prompt systems for maximum effectiveness
        combined_system_prompt = f"{enhanced_game_prompt}\n\n{system_prompt}"
        
        # Create context-aware user prompt
        enhanced_prompt = enhance_user_prompt(user_prompt, game_state, action_analysis)
        
        # Prepare messages for the AI
        messages = [
            {"role": "system", "content": combined_system_prompt},
            {"role": "user", "content": enhanced_prompt}
        ]
        
        # Add recent history for context (last 2 exchanges)
        if len(session['history']) > 0:
            recent_history = session['history'][-4:]  # Last 2 exchanges
            messages.extend(recent_history)
        
        # Add the current user message
        messages.append({"role": "user", "content": enhanced_prompt})

        # Get AI response
        response = client.chat.completions.create(
            model="llama3",
            messages=messages,
            temperature=0.8,
            max_tokens=200
        )

        ai_response = response.choices[0].message.content.strip()
        
        # Process the AI response for game mechanics (enhanced to handle new stat patterns)
        processed_response = process_ai_response_enhanced(ai_response, game_state, action_analysis)
        
        # Update session history
        session['history'].append({"role": "user", "content": user_prompt})
        session['history'].append({"role": "assistant", "content": processed_response['story']})
        session['turn_count'] = session.get('turn_count', 0) + 1
        
        # Keep history manageable (last 6 messages)
        if len(session['history']) > 6:
            session['history'] = session['history'][-6:]
        
        # Update dragon state based on health
        update_dragon_state(game_state)
        
        # Check if we should generate an image
        needs_visual = should_generate_visual(processed_response['story'])
        
        return jsonify({
            "story": processed_response['story'],
            "needs_visual": needs_visual,
            "game_effects": processed_response['effects'],
            "dragon_state": session.get('dragon_state', 'aggressive')
        })

    except Exception as e:
        return jsonify({"error": f"Generation error: {str(e)}"}), 500

def process_ai_response_enhanced(ai_response, game_state, action_analysis):
    """Enhanced AI response processing to handle the new stat patterns"""
    effects = {
        "health_change": 0,
        "mana_change": 0,
        "dragon_damage": 0,
        "items_found": [],
        "status_effects": []
    }
    
    # Enhanced pattern matching for the new stat reporting system
    # Handle "You take X damage" or "Player health -X"
    player_damage_patterns = [
        r'You take (\d+) damage',
        r'Player health -(\d+)',
        r'take (\d+) damage',
        r'suffer (\d+) damage'
    ]
    
    for pattern in player_damage_patterns:
        matches = re.findall(pattern, ai_response, re.IGNORECASE)
        for match in matches:
            effects["health_change"] -= int(match)
    
    # Handle "Dragon takes X damage" or "Dragon health -X"
    dragon_damage_patterns = [
        r'Dragon takes (\d+) damage',
        r'Dragon health -(\d+)',
        r'deals (\d+) damage to.*dragon',
        r'dragon.*takes (\d+) damage'
    ]
    
    for pattern in dragon_damage_patterns:
        matches = re.findall(pattern, ai_response, re.IGNORECASE)
        for match in matches:
            effects["dragon_damage"] += int(match)
    
    # Handle "Mana -X" or "You spend X mana"
    mana_cost_patterns = [
        r'Mana -(\d+)',
        r'You spend (\d+) mana',
        r'spend (\d+) mana',
        r'costs (\d+) mana'
    ]
    
    for pattern in mana_cost_patterns:
        matches = re.findall(pattern, ai_response, re.IGNORECASE)
        for match in matches:
            effects["mana_change"] -= int(match)
    
    # Handle "Mana +X" or "You recover X mana"
    mana_gain_patterns = [
        r'Mana \+(\d+)',
        r'You recover (\d+) mana',
        r'recover (\d+) mana',
        r'restore (\d+) mana',
        r'gain (\d+) mana'
    ]
    
    for pattern in mana_gain_patterns:
        matches = re.findall(pattern, ai_response, re.IGNORECASE)
        for match in matches:
            effects["mana_change"] += int(match)
    
    # Handle "Health restored: X" or "You heal X health"
    health_gain_patterns = [
        r'Health restored: (\d+)',
        r'You heal (\d+) health',
        r'heal (\d+) health',
        r'restore (\d+) health',
        r'gain (\d+) health'
    ]
    
    for pattern in health_gain_patterns:
        matches = re.findall(pattern, ai_response, re.IGNORECASE)
        for match in matches:
            effects["health_change"] += int(match)
    
    # Handle item discoveries "Find X item" or "Gain X item"
    item_find_patterns = [
        r'Find (\d+) (\w+)',
        r'Gain (\d+) (\w+)',
        r'discover (\d+) (\w+)',
        r'obtain (\d+) (\w+)'
    ]
    
    for pattern in item_find_patterns:
        matches = re.findall(pattern, ai_response, re.IGNORECASE)
        for amount, item in matches:
            effects["items_found"].append({
                "item": item.lower(),
                "amount": int(amount),
                "effect": "+"
            })
    
    # Keep your existing extraction logic for backward compatibility
    # Extract health changes (old format)
    health_matches = re.findall(r'HEALTH([+-])(\d+)', ai_response)
    for sign, amount in health_matches:
        change = int(amount) if sign == '+' else -int(amount)
        effects["health_change"] += change
        # Remove the command from the story
        ai_response = re.sub(f'HEALTH[+-]{amount}', '', ai_response)
    
    # Extract mana changes (old format)
    mana_matches = re.findall(r'MANA([+-])(\d+)', ai_response)
    for sign, amount in mana_matches:
        change = int(amount) if sign == '+' else -int(amount)
        effects["mana_change"] += change
        ai_response = re.sub(f'MANA[+-]{amount}', '', ai_response)
    
    # Extract dragon damage (old format)
    dragon_matches = re.findall(r'DRAGON_DAMAGE(\d+)', ai_response)
    for amount in dragon_matches:
        effects["dragon_damage"] += int(amount)
        ai_response = re.sub(f'DRAGON_DAMAGE{amount}', '', ai_response)
    
    # Extract found items (old format)
    item_matches = re.findall(r'(HEALING_POTION|MANA_ELIXIR|MAGIC_SCROLL|ANCIENT_RELIC|CRYSTAL_SHARD|FIRE_GEM|ICE_CRYSTAL|SHADOW_CLOAK|DRAGON_SCALE|MYSTIC_ORB|HOLY_WATER|CURSED_AMULET|ENCHANTED_RING|DRAGON_TOOTH)([+-])(\d+)', ai_response)
    for item, sign, amount in item_matches:
        item_name = item.lower()
        if item_name == "healing_potion":
            item_name = "potion"
        elif item_name == "mana_elixir":
            item_name = "elixir"
        
        effects["items_found"].append({
            "item": item_name,
            "amount": int(amount),
            "effect": sign
        })
        # Remove the command from the story
        ai_response = re.sub(f'{item}[+-]{amount}', '', ai_response)
    
    # Calculate combat effects based on action type (keep your existing logic)
    if action_analysis["action_type"] == "attack":
        base_damage = random.randint(15, 25)
        difficulty_modifier = {"easy": 1.2, "normal": 1.0, "hard": 0.8}[game_state.get("difficulty", "normal")]
        if effects["dragon_damage"] == 0:  # Only add if no damage was already calculated
            effects["dragon_damage"] = int(base_damage * difficulty_modifier)
        
        # Dragon counterattack
        if random.random() < DRAGON_STRATEGIES[game_state.get("difficulty", "normal")]["counter_probability"]:
            counter_damage = random.randint(10, 20)
            effects["health_change"] -= counter_damage
    
    elif action_analysis["action_type"] == "defend":
        # Defensive actions reduce incoming damage
        damage_reduction = random.randint(5, 15)
        effects["health_change"] = max(effects["health_change"], -damage_reduction)
        # Gain some mana while defending
        if effects["mana_change"] == 0:  # Only add if no mana change was already calculated
            effects["mana_change"] += random.randint(5, 10)
    
    elif action_analysis["action_type"] == "magic":
        if game_state.get("playerMana", 0) >= action_analysis["mana_cost"]:
            # Successful magic
            magic_damage = random.randint(25, 40)
            if effects["dragon_damage"] == 0:  # Only add if no damage was already calculated
                effects["dragon_damage"] = magic_damage
            if effects["mana_change"] == 0:  # Only deduct if no mana change was already calculated
                effects["mana_change"] -= action_analysis["mana_cost"]
        else:
            # Failed magic - backfire
            effects["health_change"] -= random.randint(5, 15)
            effects["mana_change"] = 0  # No mana change if failed
    
    elif action_analysis["is_creative"]:
        # Creative actions have random outcomes
        outcome = random.choice(["success", "failure", "neutral"])
        if outcome == "success":
            if effects["dragon_damage"] == 0:
                effects["dragon_damage"] = random.randint(10, 20)
            if effects["mana_change"] == 0:
                effects["mana_change"] += random.randint(5, 10)
        elif outcome == "failure":
            effects["health_change"] -= random.randint(5, 15)
        # Neutral has no effects
    
    # Clean up the AI response
    ai_response = re.sub(r'\s+', ' ', ai_response).strip()
    
    return {
        "story": ai_response,
        "effects": effects
    }

def analyze_player_action(user_prompt, game_state):
    """Analyze what the player is trying to do"""
    prompt_lower = user_prompt.lower()
    
    analysis = {
        "action_type": "unknown",
        "target": None,
        "item_used": None,
        "is_magical": False,
        "is_defensive": False,
        "is_offensive": False,
        "is_creative": False,
        "is_inappropriate": False,
        "is_vague": False,
        "mana_cost": 0,
        "risk_level": "medium"
    }
    
    # Check for inappropriate actions
    inappropriate_words = ["fuck", "sex", "rape", "molest", "seduce", "kiss", "lick", "suck"]
    if any(word in prompt_lower for word in inappropriate_words):
        analysis["is_inappropriate"] = True
        analysis["action_type"] = "inappropriate"
        return analysis
    
    # Check for vague actions
    vague_phrases = ["weak spot", "weak point", "vulnerable spot", "attack randomly", "do something"]
    if any(phrase in prompt_lower for phrase in vague_phrases):
        analysis["is_vague"] = True
        analysis["action_type"] = "vague"
    
    # Check for dance or silly actions
    if "dance" in prompt_lower or "sing" in prompt_lower or "joke" in prompt_lower:
        analysis["action_type"] = "silly"
        analysis["is_creative"] = True
    
    # Determine action type
    if any(word in prompt_lower for word in ["attack", "strike", "slash", "stab", "hit"]):
        analysis["action_type"] = "attack"
        analysis["is_offensive"] = True
    elif any(word in prompt_lower for word in ["defend", "block", "shield", "dodge", "parry"]):
        analysis["action_type"] = "defend"
        analysis["is_defensive"] = True
    elif any(word in prompt_lower for word in ["magic", "spell", "cast", "enchant", "curse"]):
        analysis["action_type"] = "magic"
        analysis["is_magical"] = True
        analysis["is_offensive"] = True
        analysis["mana_cost"] = 15
    elif any(word in prompt_lower for word in ["heal", "potion", "elixir", "drink"]):
        analysis["action_type"] = "healing"
        analysis["is_defensive"] = True
    elif any(word in prompt_lower for word in ["run", "flee", "escape", "retreat"]):
        analysis["action_type"] = "flee"
        analysis["risk_level"] = "high"
    else:
        analysis["action_type"] = "creative"
        analysis["is_creative"] = True
    
    # Check for specific items mentioned
    items = ["sword", "shield", "potion", "elixir", "scroll", "relic", "crystal", "ring", "gem"]
    for item in items:
        if item in prompt_lower:
            analysis["item_used"] = item
            break
    
    # Determine risk level
    if analysis["is_magical"] and game_state.get("playerMana", 0) < analysis["mana_cost"]:
        analysis["risk_level"] = "critical"
    elif analysis["is_creative"] or analysis["is_vague"]:
        analysis["risk_level"] = "high"
    elif analysis["is_defensive"]:
        analysis["risk_level"] = "low"
    
    return analysis

def create_dynamic_system_prompt(game_state, action_analysis):
    """Create a dynamic system prompt based on current game state"""
    difficulty = game_state.get("difficulty", "normal")
    player_health = game_state.get("playerHealth", 100)
    player_mana = game_state.get("playerMana", 50)
    dragon_health = game_state.get("dragonHealth", 150)
    dragon_state = session.get('dragon_state', 'aggressive')
    
    # Base personality
    prompt = f"""You are an epic fantasy storyteller for a dragon battle game. 

GAME STATE:
- Difficulty: {difficulty.upper()}
- Player Health: {player_health}
- Player Mana: {player_mana}
- Dragon Health: {dragon_health}
- Dragon State: {dragon_state}

STORYTELLING RULES:
1. Write vivid, dramatic responses in 2-3 sentences
2. Make the dragon's behavior match the difficulty and current state
3. Include specific health/mana changes in your narrative using format: "HEALTH+15" or "MANA-10"
4. Suggest environmental items when appropriate: "You notice a HEALING_POTION+25 glowing nearby"
5. Mock inappropriate or silly actions with humor
6. For vague actions, describe failure to find weak spots
7. Make magic fail spectacularly when mana is insufficient
8. Dragon gets more desperate and dangerous as health decreases

DRAGON BEHAVIOR ({difficulty} difficulty):"""
    
    if difficulty == "easy":
        prompt += "\n- Attacks are predictable and moderate\n- Occasionally misses or hesitates\n- Weak to most player strategies"
    elif difficulty == "normal":
        prompt += "\n- Balanced offense and defense\n- Adapts to player tactics\n- Moderate counterattacks"
    else:  # hard
        prompt += "\n- Extremely aggressive and cunning\n- Powerful special attacks\n- Punishes player mistakes severely"
    
    # Add dragon state-specific behavior
    if dragon_state == "wounded":
        prompt += "\n- Dragon is injured but more desperate\n- Uses more dangerous attacks\n- Tries to end the fight quickly"
    elif dragon_state == "desperate":
        prompt += "\n- Dragon is near death and furious\n- Uses last resort attacks\n- Becomes unpredictable and savage"
    elif dragon_state == "dying":
        prompt += "\n- Dragon is making final desperate moves\n- Extremely dangerous death throes\n- May attempt suicide attacks"
    
    return prompt

def enhance_user_prompt(user_prompt, game_state, action_analysis):
    """Enhance the user prompt with game state context"""
    enhanced = f"Player action: {user_prompt}\n\n"
    
    # Add context based on action analysis
    if action_analysis["is_inappropriate"]:
        enhanced += "CONTEXT: Player attempted inappropriate action - respond with humorous failure.\n"
    elif action_analysis["is_vague"]:
        enhanced += "CONTEXT: Player action is too vague - describe failure to find target.\n"
    elif action_analysis["action_type"] == "silly":
        enhanced += "CONTEXT: Player doing silly action - respond with comedic outcome.\n"
    elif action_analysis["is_magical"] and game_state.get("playerMana", 0) < action_analysis["mana_cost"]:
        enhanced += "CONTEXT: Player attempting magic with insufficient mana - spell fails dramatically.\n"
    
    # Add environmental context
    if game_state.get("playerHealth", 100) < 30:
        enhanced += "ENVIRONMENT: Player is badly wounded - suggest healing items nearby.\n"
    elif game_state.get("playerMana", 50) < 20:
        enhanced += "ENVIRONMENT: Player is low on mana - suggest magical items or mana sources.\n"
    
    # Add inventory context
    inventory = game_state.get("inventory", {})
    if inventory.get("magic_scroll", 0) > 0:
        enhanced += "INVENTORY: Player has magic scrolls available for powerful attacks.\n"
    
    return enhanced

def update_dragon_state(game_state):
    """Update dragon's behavioral state based on health"""
    dragon_health = game_state.get("dragonHealth", 150)
    dragon_max_health = game_state.get("dragonMaxHealth", 150)
    
    health_percentage = dragon_health / dragon_max_health
    
    if health_percentage > 0.7:
        session['dragon_state'] = "aggressive"
    elif health_percentage > 0.4:
        session['dragon_state'] = "wounded"
    elif health_percentage > 0.15:
        session['dragon_state'] = "desperate"
    else:
        session['dragon_state'] = "dying"

@app.route("/generate_scene", methods=["POST"])
def generate_scene():
    data = request.get_json()
    story_text = data.get("story", "")
    game_state = data.get("gameState", {})
    
    # Determine scene type based on story content and game state
    scene_type = detect_scene_type(story_text, game_state)
    preset = VISUAL_SCENES.get(scene_type, VISUAL_SCENES["default"])
    
    try:
        # Enhanced prompt engineering based on game state
        dragon_state = session.get('dragon_state', 'aggressive')
        difficulty = game_state.get('difficulty', 'normal')
        
        # Create dynamic visual prompt
        visual_modifiers = []
        if dragon_state == "wounded":
            visual_modifiers.append("battle-scarred dragon")
        elif dragon_state == "desperate":
            visual_modifiers.append("furious desperate dragon")
        elif dragon_state == "dying":
            visual_modifiers.append("dying dragon last stand")
        
        if difficulty == "hard":
            visual_modifiers.append("epic boss battle")
        
        modifier_text = ", ".join(visual_modifiers)
        sd_prompt = f"{preset['prompt']}, {modifier_text}, {story_text[:80]}, highly detailed, dramatic lighting, cinematic"
        
        response = requests.post(
            "http://localhost:7860/sdapi/v1/txt2img",
            json={
                "prompt": sd_prompt,
                "negative_prompt": preset["negative"],
                "width": 512,
                "height": 512,
                "steps": 20,
                "sampler_name": "DPM++ 2M Karras",
                "cfg_scale": 7
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise ValueError(f"Stable Diffusion API Error: {response.text}")
            
        image_data = response.json()["images"][0]
        return jsonify({
            "image": image_data,
            "scene_type": scene_type,
            "prompt_used": sd_prompt
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def should_generate_visual(text):
    """Smart scene detection for visual generation"""
    text = text.lower()
    visual_triggers = [
        "dragon", "breathes fire", "wings", "scales", "claws", "roars",
        "forest", "appears", "emerges", "battle", "combat", "strikes",
        "magic", "spell", "glowing", "flames", "darkness", "light"
    ]
    return any(trigger in text for trigger in visual_triggers)

def detect_scene_type(text, game_state):
    """Enhanced scene type detection based on story and game state"""
    text = text.lower()
    dragon_state = session.get('dragon_state', 'aggressive')
    
    # Priority scene detection
    if "dragon" in text:
        if dragon_state == "dying":
            return "dragon_death"
        elif dragon_state == "desperate":
            return "dragon_rage"
        else:
            return "dragon"
    
    if any(word in text for word in ["forest", "woods", "trees"]):
        return "forest"
    
    if any(word in text for word in ["castle", "tower", "fortress"]):
        return "castle"
    
    if any(word in text for word in ["magic", "spell", "enchant"]):
        return "magic"
    
    return "default"

if __name__ == "__main__":
    app.run(port=5000, debug=True)