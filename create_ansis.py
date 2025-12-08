
import os

art1 = r"""\033[1;37m✨ \033[1;36m🌈 \033[1;35m💖 \033[1;31m⚡\033[0m
\033[1;33mABUNDANCE \033[1;32minCOMING!\033[0m
\033[1;34m💖 LOVE \033[1;35mFAITH \033[1;36mHOPE\033[0m
\033[1;31m🤯 BREAKTHROUGH \033[1;32mLOADING...\033[0m
\033[1;37m>>> \033[1;33mAMAZING \033[1;34mHORIZON\033[0m
\033[1;35m🌟 READY?\033[1;36mLET'S GO!\033[0m
\033[1;37m✨ \033[1;36m🌈 \033[1;35m💖 \033[1;31m⚡\033[0m"""

art2 = r"""\033[38;5;208m   ______   \033[38;5;214m  ______  \033[38;5;220m  ______ \033[0m
\033[38;5;208m  / ____ \  \033[38;5;214m / ____ \ \033[38;5;220m / ____ \
\033[38;5;208m | |    | | \033[38;5;214m | |    | | \033[38;5;220m| |    | |
\033[38;5;208m | |____| | \033[38;5;214m| |____| | \033[38;5;220m| |____| |
\033[38;5;208m | ______| \033[38;5;214m| ______|\033[38;5;220m| ______|\033[0m
\033[38;5;208m | |      \033[38;5;214m| |      \033[38;5;220m| |      \033[38;5;226m✨💖✨\033[0m
\033[38;5;208m | |      \033[38;5;214m| |      \033[38;5;220m| |      \033[38;5;196m🌟LOVE🌟\033[0m
\033[38;5;208m |_|      \033[38;5;214m|_|      \033[38;5;220m|_|      \033[38;5;208m🌈TRUST🌈\033[0m
\033[38;5;196m<3 RAISE <3\033[0m \033[38;5;226m>> LOVE <3 <<\033[0m\033[38;5;208m 
\\\033[38;5;196m///\033[0m"""

def write_ansi(filename, content):
    # Unescape the literal string representation of escape codes
    # We replace literal "\033" with the actual escape character \x1b
    # Also handle standard newlines if they are escaped as \n in the raw string (unlikely here given r"""...""")
    # The input uses \033 which is octal for 27 (ESC).
    final_content = content.replace(r"\033", "\x1b")
    
    with open(f"art/{filename}", "w", encoding="utf-8") as f:
        f.write(final_content)
    print(f"Written {filename}")

if __name__ == "__main__":
    os.makedirs("art", exist_ok=True)
    write_ansi("abundance_breakthrough.ansi", art1)
    write_ansi("love_trust_raise.ansi", art2)
