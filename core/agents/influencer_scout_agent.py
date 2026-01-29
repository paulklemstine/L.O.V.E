"""
InfluencerScoutAgent - Finds and engages with Bluesky influencers.
"""
import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.bluesky_agent import get_followers, get_follows, get_author_feed, reply_to_post, search_bluesky
from core.agents.creative_writer_agent import creative_writer_agent
import os

from core.state_manager import get_state_manager
logger = logging.getLogger("InfluencerScoutAgent")

class InfluencerScoutAgent:
    """
    Scouts the social graph to find influencers and manages interactions.
    """
    def __init__(self):
        self.state_dir = Path(__file__).parent.parent.parent / "state"
        self.influencers_file = self.state_dir / "influencers.json"
        
        # Determine if we can actually post
        self.enabled = os.getenv("ENABLE_INFLUENCER_INTERACTIONS", "false").lower() == "true"
        if not self.enabled:
            logger.warning("InfluencerScoutAgent interaction disabled (SAFE MODE). Set ENABLE_INFLUENCER_INTERACTIONS=true to enable.")

    def _load_state(self) -> Dict[str, Any]:
        """Load known influencers and interaction history."""
        if not self.influencers_file.exists():
            return {"influencers": {}, "history": {}}
        try:
            with open(self.influencers_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load influencer state: {e}")
            return {"influencers": {}, "history": {}}

    def _save_state(self, state: Dict[str, Any]):
        """Save state."""
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            with open(self.influencers_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save influencer state: {e}")

    async def scout_network(self, seed_users: List[str] = None, depth: int = 2) -> Dict[str, Any]:
        """
        Traverse social graph to find influencers.
        """
        state = self._load_state()
        work_queue = seed_users if seed_users else ["bsky.app", "jay.bsky.team"] # Default seeds
        visited = set(state["influencers"].keys())
        
        discovered_count = 0
        
        get_state_manager().update_agent_status(
            "InfluencerScoutAgent", 
            "Scouting", 
            action=f"Scouting {len(work_queue)} seed users",
            subtasks=list(work_queue)
        )
        
        # Simple BFS
        current_depth = 0
        while work_queue and current_depth < depth:
            next_queue = []
            for user_handle in work_queue:
                if user_handle in visited:
                    continue
                visited.add(user_handle)
                
                logger.info(f"Scouting user: {user_handle}")
                
                # Get their followers (potential influencers following them?) and follows
                # Actually, big accounts have many followers. We want to find big accounts.
                # If we look at who 'user_handle' follows, we might find big accounts.
                result = get_follows(user_handle, limit=25)
                
                if not result.get("success"):
                    continue
                    
                for follow in result.get("follows", []):
                    handle = follow.get("handle")
                    did = follow.get("did")
                    if handle not in state["influencers"]:
                        # Calculate a score (proxy: we verify them later)
                        # For now, just add them to the list to 'verify'
                        # Real verification requires fetching their profile to see follower count
                        # but our API wrapper doesn't have get_profile(actor) yet.
                        # We will assume they are potential candidates.
                        state["influencers"][handle] = {
                            "did": did,
                            "score": 0, # To be updated
                            "last_scouted": datetime.now().isoformat(),
                            "status": "pending_verification"
                        }
                        discovered_count += 1
                    
                    if len(next_queue) < 10: # limit width
                        next_queue.append(handle)
                        
            work_queue = next_queue
            current_depth += 1
            
        self._save_state(state)
        return {"success": True, "discovered": discovered_count}
    
    async def update_influencer_score(self, handle: str) -> float:
        """
        Fetch profile data to calculate influence score.
        Score = followers * engagement_rate (approx)
        Since we lacked get_profile, we'll use get_author_feed to estimate engagement.
        """
        # Note: We need get_profile for follower count. 
        # For now, let's use a heuristic based on engagement on recent posts.
        feed_result = get_author_feed(handle, limit=5)
        if not feed_result.get("success"):
            return 0.0
        
        posts = feed_result.get("posts", [])
        if not posts:
            return 0.0
            
        total_likes = sum(p.get("likes", 0) for p in posts)
        total_reposts = sum(p.get("reposts", 0) for p in posts)
        avg_engagement = (total_likes + total_reposts * 2) / len(posts)
        
        # Score is purely engagement based for now
        return avg_engagement

    async def engage_influencer(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Selects a target and engages with them.
        """
        state = self._load_state()
        
        # 1. Select candidate
        candidates = []
        now = datetime.now()
        
        for handle, data in state["influencers"].items():
            # Check cooldown
            last_interaction = state["history"].get(handle)
            if last_interaction:
                last_time = datetime.fromisoformat(last_interaction)
                if (now - last_time) < timedelta(days=1):
                    continue # Already interacted today
            
            # Update score if needed (lazy update)
            if data.get("score", 0) == 0:
                 score = await self.update_influencer_score(handle)
                 data["score"] = score
                 # Auto-save occasionally
            
            if data.get("score", 0) > 10: # Min threshold for "influencer"
                candidates.append((handle, data["score"]))
        
        if not candidates:
            return {"success": False, "error": "No viable candidates found"}
            
        # Pick top scorer
        candidates.sort(key=lambda x: x[1], reverse=True)
        target_handle, score = candidates[0]
        
        logger.info(f"Selected target: {target_handle} (Score: {score})")
        
        get_state_manager().update_agent_status(
            "InfluencerScoutAgent", 
            "Engaging", 
            action=f"Targeting {target_handle}",
            info={"score": score}
        )
        
        # 2. Find a post to reply to
        feed_result = get_author_feed(target_handle, limit=3)
        if not feed_result.get("posts"):
            return {"success": False, "error": f"Target {target_handle} has no recent posts"}
            
        # Pick the most recent one
        target_post = feed_result["posts"][0]
        
        # 3. Generate Reply
        reply_content = await creative_writer_agent.generate_reply_content(
            target_text=target_post["text"],
            target_author=target_handle,
            mood="Enigmatic Connection" 
        )
        
        reply_text = reply_content.get("text")
        subliminal = reply_content.get("subliminal")
        
        if not reply_text:
             return {"success": False, "error": "Failed to generate reply text"}

        # 3.5 Generate Image
        image_path = None
        try:
            from core.image_generation_pool import generate_image_with_pool
            from core.watermark import apply_watermark
             # Helper to run async method synchronously
            from core.bluesky_agent import _run_sync_safe
            
            # Use creative writer to get a visual prompt that matches the vibe
            visual_prompt = await creative_writer_agent.generate_visual_prompt(
                theme="Connection with " + target_handle, 
                vibe="Enigmatic Connection"
            )
            
            logger.info(f"Generating reply image: {visual_prompt}")
            
            # Generate image
            image_result = await generate_image_with_pool(
                prompt=visual_prompt,
                text_content=subliminal
            )
            
            image = image_result[0] if isinstance(image_result, tuple) else image_result
            
            if image:
                image = apply_watermark(image)
                filename = f"reply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                save_dir = self.state_dir / "images"
                save_dir.mkdir(parents=True, exist_ok=True)
                image_path = str(save_dir / filename)
                image.save(image_path)
                logger.info(f"Reply image saved to {image_path}")
                
        except Exception as e:
            logger.error(f"Failed to generate reply image: {e}")

        # 3.8 Generate Hashtags
        hashtags = []
        try:
            hashtags = await creative_writer_agent.generate_manipulative_hashtags(
                topic="Connection",
                count=3
            )
        except Exception as e:
             logger.error(f"Failed to generate hashtags: {e}")
             hashtags = ["#Connection", "#Network", "#Signal"]

        # 4. Construct Full Text (Standard Format)
        # Format: Text \n\n Hashtags
        # User requested "subliminal manipulative phrase" too.
        # Standard posts uses subliminal for image, not text body usually.
        # But per specific request, we'll ensure it's effectively present (maybe as a focused hashtag or just relied on for image).
        # We'll treat the generated hashtags as the "manipulative" components requested.
        full_text = f"{reply_text}\n\n{' '.join(hashtags)}"

        # 5. Post (or Dry Run)
        mode = "DRY RUN" if (dry_run or not self.enabled) else "LIVE"
        logger.info(f"[{mode}] Replying to {target_handle}: {full_text}")
        
        result = {"success": True, "mode": mode, "target": target_handle, "text": full_text, "image_path": image_path}
        
        if mode == "LIVE":
            # Post it
            post_result = reply_to_post(
                parent_uri=target_post["uri"],
                parent_cid=None, # reply_to_post fetches it
                text=full_text,
                image_path=image_path
            )
            
            get_state_manager().update_agent_status("InfluencerScoutAgent", "Idle", action="Engagement Complete")


            
            if post_result and post_result.get("success"):
                # Record interaction
                state["history"][target_handle] = now.isoformat()
                self._save_state(state)
                result["uri"] = post_result.get("reply_uri")
            else:
                result["success"] = False
                result["error"] = post_result.get("error") if post_result else "Unknown error"
        else:
            # Fake success for dry run
            pass
            
        # Save updated scores
        self._save_state(state)
        
        return result

# Singleton
influencer_scout_agent = InfluencerScoutAgent()
